import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
sys.path.append("/home/avc6555/research/MedH/Mitigation/LVLMs/llava-med-1.5")

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from llava.model.moe_llava import LoRA_MOE_FFN, LoRA_MOE_QK, LoRA_MOE_QK_old

from PIL import Image, ImageFile, UnidentifiedImageError

# Shared MedHEval input contract: tolerate JPEGs missing terminal bytes while
# still failing on empty or unidentifiable files.
ImageFile.LOAD_TRUNCATED_IMAGES = True
import math
from transformers import set_seed, logging
from corrected_sgta.visual_token_ranges import expanded_visual_range
from corrected_sgta.generation_trace import classify_generated_tokens

logging.set_verbosity_error()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def generation_budget(args, official_default):
    return official_default if args.max_new_tokens is None else args.max_new_tokens


def deterministic_process_images(images, image_processor, model_cfg):
    """Match the canonical adapter without upstream random one-pixel padding."""

    processed = []
    pad = getattr(model_cfg, "image_aspect_ratio", None) == "pad"
    mean = tuple(int(value * 255) for value in image_processor.image_mean)
    for source in images:
        image = source.convert("RGB")
        if pad and image.width != image.height:
            side = max(image.size)
            canvas = Image.new("RGB", (side, side), mean)
            canvas.paste(
                image,
                ((side - image.width) // 2, (side - image.height) // 2),
            )
            image = canvas
        processed.append(
            image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        )
    if all(item.shape == processed[0].shape for item in processed):
        return torch.stack(processed, dim=0)
    return processed


def generated_suffix_ids(sequences, tokenizer):
    """Remove generation-only boundary specials from a decoded suffix."""

    return classify_generated_tokens(
        sequences[0].tolist(),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=None,
    )["generated_token_ids"]


def image_key_position(input_ids, model):
    vision_tower = model.get_vision_tower()
    num_image_tokens = int(getattr(vision_tower, "num_patches", 0))
    return expanded_visual_range(
        input_ids,
        image_token_index=IMAGE_TOKEN_INDEX,
        num_image_tokens=num_image_tokens,
    )


def eval_model(args):
    set_seed(args.seed)
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    # model_base = os.path.expanduser(args.model_base)
    # model_name = get_model_name_from_path(model_path)
    model_name = "mistral_llava_med_1.5"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, peft_path=args.peft_path)

    if args.baseline == "VCD":
        from VCD_files.vcd_sample import evolve_vcd_sampling
        evolve_vcd_sampling()
    elif args.baseline == "avisc" or args.baseline == "m3id" or args.baseline == "damro":
        print(f"use and set {args.baseline} sampling")
        from avisc_utils.vcd_add_noise import add_diffusion_noise
        if args.baseline == "avisc":
            from corrected_sgta.avisc_sample_dynamic import evolve_avisc_sampling
        else:
            from avisc_utils.avisc_sample import evolve_avisc_sampling
        evolve_avisc_sampling()
    

    #LOAD LORA
    peft_path = args.peft_path
    if peft_path is not None and len(str(peft_path))>4:
        from peft import PeftModel
        print(f"Loading LoRA weights from {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        print(f"Merging weights")
        model = model.merge_and_unload()
        
        moe_path = os.path.join(peft_path, "non_lora_trainables.bin")
        moe_state_dict = torch.load(moe_path, map_location='cuda')
        if len(moe_state_dict.keys()) > 32:
            print("load MoE parameters!")
            top_layers = []
            num_layers = len(model.base_model.layers)
            for i in range(num_layers):
                if i not in top_layers:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_experts=args.q_expert_num,
                            original_module=original_q,
                            dense_moe=args.dense_moe).bfloat16()
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK_old(args=None,
                            lora_rank=args.lora_r,
                            lora_alpha=args.lora_alpha,
                            num_experts=args.k_expert_num,
                            top_moe_experts=args.top_moe_num,
                            original_module=original_k).bfloat16()
                else:
                    original_q = model.base_model.layers[i].self_attn.q_proj
                    model.base_model.layers[i].self_attn.q_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_q,
                            dense_moe=args.dense_moe).bfloat16()
                    original_k = model.base_model.layers[i].self_attn.k_proj
                    model.base_model.layers[i].self_attn.k_proj = \
                        LoRA_MOE_QK(args=None,
                            lora_rank=args.lora_r * 4,
                            lora_alpha=args.lora_alpha * 4,
                            num_experts=1,
                            original_module=original_k).bfloat16()
            
                
            new_state_dict = {}
            for key, value in moe_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
                # new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict, strict=False)
            model = model.to("cuda")
            for key in new_state_dict.keys():
                # if "mm_projector" in key:
                #     continue
                assert torch.equal(model.state_dict()[key], new_state_dict[key]), f"Mismatch in {key}, {model.state_dict()[key].dtype}, {new_state_dict[key].dtype}"
            print("Subset loaded successfully.")
            print('Convert to FP16...')
            model.to(torch.float16)
        elif len(moe_state_dict.keys()) > 1:
            new_state_dict = {}
            for key, value in moe_state_dict.items():
                # Replace "base_model.model" with an empty string to remove it
                new_key = key.replace("base_model.model", "")
                if new_key.startswith("."):
                    new_key = new_key[1:]
                new_state_dict[new_key] = value.to("cuda")
            model.load_state_dict(new_state_dict, strict=False)
            print("Projector loaded successfully.")
            model = model.to("cuda")
            # for key in new_state_dict.keys():
            #     print(model.state_dict()[key], "cernijvj" ,new_state_dict[key])
            #     assert torch.equal(model.state_dict()[key], new_state_dict[key]), f"Mismatch in {key}"
            model.to(torch.float16)

    if args.baseline == "PAI":
        from transformers.generation.logits_process import LogitsProcessorList
        from PAI_files.model_loader import init_cfg_processor
        from PAI_files.attention import llama_modify
        llama_modify(
            model,
            start_layer=2,
            end_layer=32,
            alpha=0.2,
            use_attn=True,
            use_cfg=True,
            # The true range is installed per sample after tokenization.
            img_start_idx=0,
            img_end_idx=0
        )

    # questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):

        if "conversations" in line:
            question = line["conversations"][0] # ['value'].split('\n')[0]
            gt_ans = line["conversations"][1]['value'] # ['value']        
            qs = question['value']
        else:
            question = line['question']
            # Formal generation manifests are label-redacted.  References are
            # joined only by the downstream evaluator and must not be required
            # or copied into raw generation artifacts.
            gt_ans = line.get('answer')
            qs = question
            if line.get("prompt_contract") != "anchor-ce-v1":
                if 'choices' in line and len(line['choices']) > 10:
                    qs += " Please choose from the following options: " + line['choices']
                if 'question_type' in line and line['question_type'] == 'binary':
                    qs += " Please answer Yes or No."

        if "qid" in line:
            idx = line["qid"]
        else:
            idx = line["id"]
        # question = line["conversations"][0] # ['value'].split('\n')[0]
        # gt_ans = line["conversations"][1] # ['value']      
        # image_file = line["image"]
        # assert gt_ans['from'] == 'gpt'
        if 'image' in line:
            image_file = line["image"]
        elif 'img_name' in line:
            image_file = line["img_name"]

        # qs = question['value']
        qs = qs.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        visual_range = image_key_position(input_ids, model)

        try:
            image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            answer_record = {
                "question_id": idx,
                "prompt": cur_prompt,
                "text": "",
                "model_id": model_name,
                "metadata": {
                    "generated_token_ids": [],
                    "raw_generated_token_ids": [],
                    "raw_generated_token_count": 0,
                    "decoded_sequence_token_count": 0,
                    "stop_reason": "input_unavailable",
                    "input_error": f"{type(exc).__name__}: {exc}",
                    "keyword_stopping_enabled": not args.disable_keyword_stopping,
                },
            }
            if gt_ans is not None:
                answer_record["gt_ans"] = gt_ans
            ans_file.write(json.dumps(answer_record) + "\n")
            ans_file.flush()
            continue
        image_tensor = deterministic_process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        stopping_criteria_arg = [] if args.disable_keyword_stopping else [stopping_criteria]


        if args.baseline == "opera":
            key_position = image_key_position(input_ids, model)
            # print("use opera")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    num_beams=5,
                    do_sample=False,
                    max_new_tokens=generation_budget(args, 128),
                    output_attentions=True,
                    opera_decoding=True,
                    key_position=key_position,
                    scale_factor=25,
                    threshold=25,
                    num_attn_candidates=5,
                    penalty_weights=1,
                    stopping_criteria=stopping_criteria_arg)
        elif args.baseline == "PAI":
            for layer_index in range(2, 32):
                attention = model.model.layers[layer_index].self_attn
                attention.img_start_idx = visual_range["image_start"]
                attention.img_end_idx = visual_range["image_end"] + 1
            logits_processor = (
                    init_cfg_processor(tokenizer=tokenizer, llm_model=model, questions=[cur_prompt], gamma=1.1, beam=1, start_layer=2, end_layer=32)
                )
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=generation_budget(args, 1024),
                    output_attentions=False,
                    output_hidden_states=False,
                    logits_processor=LogitsProcessorList([logits_processor]),
                    stopping_criteria=stopping_criteria_arg)
        
        elif args.baseline == "VCD":
            # print("use VCD")
            from VCD_files.vcd_add_noise import add_diffusion_noise
            image_tensor_cd = add_diffusion_noise(image_tensor, 500) #args.noise_step
            # image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                        cd_alpha = 1, #args.cd_alpha,
                        cd_beta = 0.1, #args.cd_beta,
                        do_sample=True,
                        max_new_tokens=generation_budget(args, 1024),
                        temperature=1,
                        output_attentions=False,
                        output_hidden_states=False,
                        stopping_criteria=stopping_criteria_arg
                        )
        elif args.baseline == "avisc":
            use_cd = False
            if use_cd:
                image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
            else:
                image_tensor_cd = None
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        images_cd=(image_tensor_cd.half().cuda() if image_tensor_cd is not None else None),
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=generation_budget(args, 1024),
                        use_avisc=True,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        model_name="llava",
                        avisc_image_start=image_key_position(input_ids, model)["image_start"],
                        avisc_num_image_tokens=image_key_position(input_ids, model)["num_image_tokens"],
                        stopping_criteria=stopping_criteria_arg
                    )    
        elif args.baseline == "m3id":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        images_cd=None,
                        cd_alpha=1.0,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=generation_budget(args, 1024),
                        use_avisc=False,
                        use_m3id=True,
                        layer_gamma=0.5,
                        lamb=1.0,
                        avisc_image_start=visual_range["image_start"],
                        avisc_num_image_tokens=visual_range["num_image_tokens"],
                        stopping_criteria=stopping_criteria_arg
                    )    
        elif args.baseline == "DoLa":
            with torch.no_grad():
                early_exit_layers = [0,2,4,6,8,10,12,14,32]
                mature_layer = early_exit_layers[-1]
                premature_layer = None
                candidate_premature_layers = early_exit_layers[:-1]
                output_ids = model.generate(input_ids,
                                            attention_mask=attention_mask,
                                            images=image_tensor.unsqueeze(0).half().cuda(),
                                            image_sizes=[image.size],
                                            max_new_tokens=generation_budget(args, 1024),
                                            dola_decoding=True,
                                            top_p=0.95, top_k=0, temperature=0.9, 
                                            stopping_criteria=stopping_criteria_arg, relative_top=0.1,
                                            mature_layer=mature_layer, premature_layer=None, candidate_premature_layers=candidate_premature_layers,
                                            )
        elif args.baseline == "damro":
            with torch.inference_mode():
                output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        images_cd=None,
                        cd_alpha=0.5,
                        cd_beta=0.1,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1,
                        top_k=None,
                        max_new_tokens=generation_budget(args, 1024),
                        use_damro=True,
                        use_avisc=False,
                        layer_gamma=0.5,
                        masking_scheme="zeros",
                        lamb=1.0,
                        temp=1.0,
                        avisc_image_start=visual_range["image_start"],
                        avisc_num_image_tokens=visual_range["num_image_tokens"],
                        stopping_criteria=stopping_criteria_arg
                    )    
        elif args.baseline in {
            "VISTA", "VISTA_off", "VISTA_VSV", "VISTA_SLA"
        }:
            from corrected_sgta.vista_adapter import VistaRuntimeAdapter

            vista_enabled = args.baseline != "VISTA_off"
            vista_vsv_enabled = args.baseline in {"VISTA", "VISTA_VSV"}
            vista_sla_enabled = args.baseline in {"VISTA", "VISTA_SLA"}
            negative_input_ids = input_ids[
                input_ids.ne(IMAGE_TOKEN_INDEX)
            ].reshape(input_ids.shape[0], -1)
            negative_kwargs = {
                "input_ids": negative_input_ids,
                "attention_mask": torch.ones_like(negative_input_ids),
                "images": None,
            }
            positive_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "images": image_tensor.unsqueeze(0).half().cuda(),
                "image_sizes": [image.size],
            }
            with VistaRuntimeAdapter(
                model=model,
                enabled=vista_enabled,
                enable_vsv=vista_vsv_enabled,
                enable_sla=vista_sla_enabled,
                negative_kwargs=negative_kwargs,
                positive_kwargs=positive_kwargs,
                vsv_lambda=args.vista_vsv_lambda,
                vsv_layers=args.vista_vsv_layers,
                logits_layers=args.vista_logits_layers,
                logits_alpha=args.vista_logits_alpha,
            ):
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[image.size],
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=generation_budget(args, 1024),
                        stopping_criteria=stopping_criteria_arg,
                    )
        elif args.baseline == "beam":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=False,
                    num_beams=5,
                    max_new_tokens=generation_budget(args, 1024),
                    stopping_criteria=stopping_criteria_arg)
        elif args.baseline == "greedy":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=generation_budget(args, 1024),
                    stopping_criteria=stopping_criteria_arg)
        elif args.baseline == "nucleus":
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True,
                    top_p=0.9,                 # Cumulative probability threshold
                    top_k=0,                   # Optional; setting to 0 disables top-k filtering
                    max_new_tokens=generation_budget(args, 1024),
                    stopping_criteria=stopping_criteria_arg)
        else:

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    # do_sample=False,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # num_beams=5,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=generation_budget(args, 1024),
                    # stopping_criteria=stopping_criteria,
                    use_cache=True)

        sequences = output_ids.sequences if hasattr(output_ids, "sequences") else output_ids
        raw_sequence_length = int(sequences.shape[1])
        prompt_prefix_in_sequence = False
        if (
            sequences.ndim == 2
            and sequences.shape[1] > input_ids.shape[1]
            and torch.equal(sequences[:, : input_ids.shape[1]], input_ids)
        ):
            prompt_prefix_in_sequence = True
            sequences = sequences[:, input_ids.shape[1] :]
        generation_trace = classify_generated_tokens(
            sequences[0].tolist(),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
        )
        generated_ids = generation_trace["generated_token_ids"]
        outputs = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # ans_id = shortuuid.uuid()
        answer_record = {"question_id": idx,
                         "prompt": cur_prompt,
                         "text": outputs,
                         # "answer_id": ans_id,
                         "model_id": model_name,
                         "metadata": {
                             "input_token_count": int(input_ids.shape[1]),
                             "raw_sequence_token_count": raw_sequence_length,
                             "decoded_sequence_token_count": len(generated_ids),
                             "generated_token_ids": generated_ids,
                             "raw_generated_token_ids": generation_trace["raw_generated_token_ids"],
                             "raw_generated_token_count": generation_trace["raw_generated_token_count"],
                             "terminal_token_ids": generation_trace["terminal_token_ids"],
                             "stop_reason": generation_trace["stop_reason"],
                             "prompt_prefix_in_sequence": prompt_prefix_in_sequence,
                             "keyword_stopping_enabled": not args.disable_keyword_stopping,
                         }}
        if gt_ans is not None:
            answer_record["gt_ans"] = gt_ans
        ans_file.write(json.dumps(answer_record) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--peft-path", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=0)
    parser.add_argument("--k_expert_num", type=int, default=0)
    parser.add_argument("--q_expert_num", type=int, default=0)
    parser.add_argument("--dense_moe", type=bool, default=False)
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--top_moe_num", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--disable-keyword-stopping", action="store_true")
    parser.add_argument("--vista-vsv-lambda", type=float, default=0.01)
    parser.add_argument("--vista-vsv-layers", type=str, default=None)
    parser.add_argument("--vista-logits-layers", type=str, default="25,30")
    parser.add_argument("--vista-logits-alpha", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eval_model(args)
