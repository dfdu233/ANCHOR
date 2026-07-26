import {
  ArrowDown,
  ArrowUpRight,
  Check,
  CircleAlert,
  Code2,
  Database,
  Github,
  Image as ImageIcon,
  Layers3,
  Microscope,
  ShieldCheck,
  TriangleAlert,
  X,
} from "lucide-react";

const repo = "https://github.com/dfdu233/ANCHOR";

const cxrResults = [
  { label: "Complete-sequence identity", value: 70.78, tone: "muted" },
  { label: "RULE greedy / POPE", value: 75.56, tone: "muted" },
  { label: "Source-margin calibration", value: 79.25, tone: "accent" },
];

const reportResults = [
  { dataset: "IU-Xray", baseline: 9.54, method: 14.46, delta: "+4.92" },
  { dataset: "MIMIC-CXR", baseline: 10.52, method: 17.07, delta: "+6.55" },
  { dataset: "Harvard*", baseline: 12.86, method: 12.46, delta: "−0.40" },
];

const probeResults = [
  { task: "Context", base: 86.72, oracle: 89.84, sgta: 86.72 },
  { task: "CXR", base: 60.94, oracle: 63.28, sgta: 60.94 },
  { task: "Knowledge", base: 60.94, oracle: 67.19, sgta: 60.94 },
  { task: "Multimodal", base: 57.03, oracle: 65.62, sgta: 57.03 },
];

const scatResults = [
  { task: "LLaVA · CXR", baseline: 46.35, scat: 73.05, delta: "+26.70" },
  { task: "LLaVA · MM", baseline: 42.29, scat: 65.64, delta: "+23.35" },
  { task: "LLaVA · Knowledge", baseline: 33.16, scat: 63.16, delta: "+30.00" },
  { task: "Hulu · CXR", baseline: 63.00, scat: 74.11, delta: "+11.11" },
];

const methodFamilies = [
  ["SCA-T", "Transductive CP", "Full cache", "Semantic class prototypes with TIM/TIM-KL adaptation over calibration and test features.", "https://papers.miccai.org/miccai-2025/0955-Paper4783.html"],
  ["LAME / LATA", "Graph adaptation", "Integrated", "Laplacian label assignment; LATA extends the graph view to conformal medical-VLM adaptation.", "https://openaccess.thecvf.com/content/CVPR2026/papers/Bozorgtabar_LATA_Laplacian-Assisted_Transductive_Adaptation_for_Conformal_Uncertainty_in_Medical_VLMs_CVPR_2026_paper.pdf"],
  ["VCD", "Visual contrast", "Pilot cache", "Contrasts next-token distributions from the original and a distorted image.", "https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html"],
  ["DoLa", "Layer contrast", "Pilot cache", "Contrasts mature and premature language-layer logits to surface factual knowledge.", "https://proceedings.iclr.cc/paper_files/paper/2024/hash/edc36117f795ca52a0cbf6a7b3882859-Abstract-Conference.html"],
  ["OPERA", "Attention decoding", "Integrated", "Penalizes over-trusted summary tokens and retrospectively reallocates beam search.", "https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.pdf"],
  ["M3ID", "Mutual information", "Integrated", "Amplifies the visual prompt through multimodal mutual-information decoding.", "https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf"],
  ["DAMRO", "Token suppression", "Integrated", "Uses ViT CLS attention to identify and suppress background outlier tokens.", "https://aclanthology.org/2024.emnlp-main.439/"],
  ["PAI", "Image attention", "Integrated", "Reweights image-token attention at inference to counter linguistic over-reliance.", "https://arxiv.org/abs/2407.21771"],
  ["VISTA", "Visual steering", "Integrated", "Combines visual activation steering with early-layer token-logit augmentation.", "https://arxiv.org/abs/2502.03628"],
  ["AvisC", "Information flow", "Integrated", "Adaptively constrains irrelevant visual information as it enters the language model.", "https://ojs.aaai.org/index.php/AAAI/article/view/34512"],
];

function Tag({ children, tone = "neutral" }) {
  return <span className={`tag tag-${tone}`}>{children}</span>;
}

function Metric({ value, label, note }) {
  return (
    <article className="metric">
      <strong>{value}</strong>
      <span>{label}</span>
      <small>{note}</small>
    </article>
  );
}

function Bar({ value, max = 85, tone = "accent" }) {
  return (
    <div className="bar-track" aria-label={`${value}%`}>
      <div className={`bar-fill ${tone}`} style={{ width: `${(value / max) * 100}%` }} />
    </div>
  );
}

function EvidenceBadge({ status }) {
  const map = {
    verified: [Check, "Verified artifact", "good"],
    pilot: [CircleAlert, "Pilot / source validation", "warn"],
    negative: [X, "Did not pass gate", "bad"],
  };
  const [Icon, label, tone] = map[status];
  return (
    <span className={`evidence-badge ${tone}`}>
      <Icon size={14} /> {label}
    </span>
  );
}

export default function Page() {
  return (
    <main>
      <nav className="nav">
        <a className="brand" href="#top" aria-label="ANCHOR home">
          <span className="brand-mark">A</span>
          <span>ANCHOR</span>
        </a>
        <div className="nav-links">
          <a href="#method">Method</a>
          <a href="#evidence">Evidence</a>
          <a href="#baselines">Baselines</a>
          <a href="#cases">Cases</a>
          <a href="#limits">Limits</a>
        </div>
        <a className="button button-small" href={repo} target="_blank" rel="noreferrer">
          <Github size={16} /> Code
        </a>
      </nav>

      <section className="hero section" id="top">
        <div className="eyebrow">Reliable medical vision–language models under visual domain shift</div>
        <h1>
          What if the same clinical evidence
          <br />
          <em>looks native to another hospital?</em>
        </h1>
        <p className="hero-copy">
          ANCHOR studies hallucination through the lens of domain generalization. It asks whether
          source-guided visual interventions and sequence-level evidence alignment can preserve
          clinical content while reducing acquisition-specific bias.
        </p>
        <div className="hero-actions">
          <a className="button" href="#evidence">
            Explore the evidence <ArrowDown size={17} />
          </a>
          <a className="button button-ghost" href={`${repo}#quick-start`} target="_blank" rel="noreferrer">
            Reproduce locally <Code2 size={17} />
          </a>
        </div>
        <div className="scope-strip">
          <div><Database size={17} /><span>IU-Xray · MIMIC-CXR · SLAKE · VQA-RAD</span></div>
          <div><Microscope size={17} /><span>LLaVA-Med · Hulu-Med</span></div>
          <div><ShieldCheck size={17} /><span>Target-free selection where stated</span></div>
        </div>
      </section>

      <section className="section motivation">
        <div className="section-heading compact">
          <div>
            <span className="kicker">01 / Motivation</span>
            <h2>One radiograph, multiple acquisition styles.</h2>
          </div>
          <p>
            Low-frequency source centers change appearance while retaining phase and the DC bin.
            These are mechanism views—not evidence that the final gate succeeded.
          </p>
        </div>
        <div className="view-grid">
          {[
            ["Original", "/views/original.png", "Input evidence"],
            ["IU-Xray center", "/views/rule_iuxray.png", "ρ .02 · β .50"],
            ["SLAKE center", "/views/slake_xray.png", "ρ .02 · β .50"],
            ["VQA-RAD center", "/views/vqa_rad_train.png", "ρ .02 · β .50"],
          ].map(([title, src, detail], index) => (
            <figure className={`view-card ${index === 0 ? "original" : ""}`} key={title}>
              <div className="image-shell">
                <img src={src} alt={`${title} view of the same chest radiograph`} />
                <span className="image-index">0{index + 1}</span>
              </div>
              <figcaption>
                <strong>{title}</strong>
                <span>{detail}</span>
              </figcaption>
            </figure>
          ))}
        </div>
        <div className="motivation-note">
          <TriangleAlert size={20} />
          <p>
            The cleaned source classifier distinguished the three style sources at <b>84.7%</b>,
            but none of nine tested transforms moved images reliably toward the requested source.
            The visualization motivates the question; it does not hide the failed intervention gate.
          </p>
        </div>
      </section>

      <section className="section method-section" id="method">
        <div className="section-heading">
          <div>
            <span className="kicker">02 / Method lens</span>
            <h2>From source statistics to invariant evidence.</h2>
          </div>
          <p>
            The research path is intentionally compact: define source style, create a
            content-preserving counterfactual, align token evidence, and keep inference single-pass.
          </p>
        </div>
        <div className="method-flow">
          {[
            [Database, "Source bank", "Robust log-spectrum centers from source-only chest radiographs."],
            [ImageIcon, "Counterfactual view", "Same clinical content, altered acquisition style."],
            [Layers3, "Evidence objective", "Complete-sequence risk across the source orbit."],
            [ArrowUpRight, "Single-pass inference", "One original image; no target-domain access."],
          ].map(([Icon, title, copy], index) => (
            <article className="method-step" key={title}>
              <div className="method-number">{index + 1}</div>
              <Icon size={24} />
              <h3>{title}</h3>
              <p>{copy}</p>
            </article>
          ))}
        </div>
        <div className="equation">
          <div>
            <span>Source-Orbit robust risk</span>
            <code>ℒ = log[(exp ℓ₀ + exp ℓ₊) / 2]</code>
          </div>
          <p>
            No extra loss weight. When the view equals the input, the objective reduces exactly to
            task-only sequence NLL.
          </p>
        </div>
      </section>

      <section className="section evidence-section" id="evidence">
        <div className="section-heading">
          <div>
            <span className="kicker">03 / Evidence dashboard</span>
            <h2>Results, with their scope attached.</h2>
          </div>
          <p>
            Every panel is tied to a cached artifact. Full results and exploratory diagnostics are
            kept visually distinct.
          </p>
        </div>

        <div className="headline-metrics">
          <Metric value="+3.69 pp" label="MIMIC-CXR vs RULE greedy" note="n=3,470 · 218 patients" />
          <Metric value="+9.38 pp" label="IU-Xray source validation" note="n=384 · CI [4.43, 14.32]" />
          <Metric value="+6.55" label="MIMIC ROUGE-L points" note="report generation · n=694" />
          <Metric value="+0.32 pp" label="Orbit LODO macro" note="failed the preregistered 3pp gate" />
        </div>

        <div className="evidence-grid">
          <article className="panel panel-wide">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="verified" />
                <h3>Full MIMIC closed-ended evaluation</h3>
              </div>
              <Tag>3,470 questions</Tag>
            </div>
            <div className="bar-chart">
              {cxrResults.map((item) => (
                <div className="bar-row" key={item.label}>
                  <div className="bar-label"><span>{item.label}</span><b>{item.value.toFixed(2)}%</b></div>
                  <Bar value={item.value} tone={item.tone} />
                </div>
              ))}
            </div>
            <div className="stat-line">
              <span><b>+3.69 pp</b> vs RULE greedy</span>
              <span>patient-bootstrap 95% CI <b>[2.34, 5.01]</b></span>
              <span>McNemar <b>p = 2.05×10⁻⁸</b></span>
            </div>
            <p className="caption">
              Source-margin calibration was frozen before target evaluation. This is strong
              sequence-interface evidence, not yet an independent proof of DG.
            </p>
          </article>

          <article className="panel">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="pilot" />
                <h3>Where rescues happen</h3>
              </div>
              <Tag>796 flips</Tag>
            </div>
            <div className="rescue-list">
              {[
                ["Effusion", 130, 7],
                ["Opacity / infiltrate", 100, 14],
                ["Normal / clear", 60, 2],
                ["Cardiac", 13, 63],
                ["Device / line", 9, 19],
              ].map(([name, rescue, harm]) => (
                <div className="rescue-row" key={name}>
                  <span>{name}</span>
                  <div className="splitbar">
                    <i className="rescue" style={{ width: `${Math.min(100, rescue / 1.3)}%` }} />
                    <i className="harm" style={{ width: `${Math.min(100, harm / 0.63)}%` }} />
                  </div>
                  <small><b>{rescue}</b> / {harm}</small>
                </div>
              ))}
            </div>
            <p className="legend"><i className="dot rescue" /> rescue <i className="dot harm" /> harm</p>
          </article>

          <article className="panel">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="verified" />
                <h3>Report generation</h3>
              </div>
              <Tag>ROUGE-L ×100</Tag>
            </div>
            <div className="report-chart">
              {reportResults.map((r) => (
                <div className="report-row" key={r.dataset}>
                  <span>{r.dataset}</span>
                  <div className="paired-bars">
                    <i className="baseline" style={{ width: `${r.baseline * 4}%` }} />
                    <i className="method" style={{ width: `${r.method * 4}%` }} />
                  </div>
                  <b className={r.delta.startsWith("−") ? "negative" : ""}>{r.delta}</b>
                </div>
              ))}
            </div>
            <p className="caption">
              Overall n=1,997: ROUGE-L 11.07→14.65. *Harvard is retained as an OOD stress test,
              not chest-radiograph institutional evidence.
            </p>
          </article>

          <article className="panel panel-wide">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="pilot" />
                <h3>Training-free SGTA probe: oracle headroom vs realized gain</h3>
              </div>
              <Tag>Hulu-Med · n=128/task</Tag>
            </div>
            <div className="probe-table">
              <div className="probe-row header"><span>Task</span><span>Baseline</span><span>Style oracle</span><span>SGTA-v3</span></div>
              {probeResults.map((r) => (
                <div className="probe-row" key={r.task}>
                  <b>{r.task}</b>
                  <span>{r.base.toFixed(2)}%</span>
                  <span className="oracle">{r.oracle.toFixed(2)}%</span>
                  <span>{r.sgta.toFixed(2)}%</span>
                </div>
              ))}
            </div>
            <p className="caption">
              Different views contain recoverable oracle headroom, especially on multimodal and
              knowledge tasks, but the conservative selector chose the original prediction in all
              eight Hulu/LLaVA endpoints. Potential is not realized improvement.
            </p>
          </article>

          <article className="panel negative-panel">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="negative" />
                <h3>ANCHOR-Orbit LODO</h3>
              </div>
              <Tag tone="bad">Gate failed</Tag>
            </div>
            <div className="orbit-number">+0.32<small>pp macro</small></div>
            <p>
              Task-only 43.26% → Orbit 43.58%. The smooth worst-view correction was only
              1.6×10⁻⁴–3.3×10⁻⁴ and produced the same text as source-average on all 299 samples.
            </p>
          </article>

          <article className="panel negative-panel">
            <div className="panel-head">
              <div>
                <EvidenceBadge status="negative" />
                <h3>SGTA-ConfGen pilot</h3>
              </div>
              <Tag tone="bad">Vacuous</Tag>
            </div>
            <div className="set-visual" aria-label="all four candidates retained">
              {[1, 2, 3, 4].map((i) => <span key={i}>{i}</span>)}
              <b>λ = ∞</b>
            </div>
            <p>
              All four open-ended tasks returned every candidate. Coverage without useful set
              reduction is not presented as a positive conformal result.
            </p>
          </article>
        </div>
      </section>


      <section className="section baseline-section" id="baselines">
        <div className="section-heading">
          <div><span className="kicker">04 / Baseline landscape</span><h2>Strong adaptation, decoding, and attention baselines.</h2></div>
          <p>SCA-T has a completed fixed-class cache. VCD and DoLa have n=32 activation pilots. The remaining methods are integrated from upstream code but do not yet have protocol-complete full results.</p>
        </div>
        <div className="scat-block">
          <div className="scat-intro">
            <EvidenceBadge status="verified" />
            <h3>SCA-T is the strongest closed-ended comparator</h3>
            <p>It replaces fragile surface-token logits with Yes/No semantic prototypes, then performs full-batch TIM/TIM-KL adaptation. The gains are large, but they are transductive fixed-class results—not open-ended generation and not SGTA gains.</p>
            <a href="https://papers.miccai.org/miccai-2025/0955-Paper4783.html" target="_blank" rel="noreferrer">MICCAI 2025 paper <ArrowUpRight size={14} /></a>
          </div>
          <div className="scat-table">
            <div className="scat-row scat-header"><span>Endpoint</span><span>Surface</span><span>SCA-T</span><span>Δ pp</span></div>
            {scatResults.map((row) => <div className="scat-row" key={row.task}><b>{row.task}</b><span>{row.baseline.toFixed(2)}</span><span className="scat-score">{row.scat.toFixed(2)}</span><strong>{row.delta}</strong></div>)}
          </div>
        </div>
        <div className="pilot-strip">
          <div><span>MIMIC-CXR · n=32 activation pilot</span><b>Greedy 71.88</b></div>
          <div><span>VCD</span><b className="positive">78.13</b><small>+6.25 pp</small></div>
          <div><span>DoLa</span><b>71.88</b><small>+0.00 pp</small></div>
          <p>Protocol reconstruction is not paper-exact; pilot values are not promoted to the main result table.</p>
        </div>
        <div className="method-family-grid">
          {methodFamilies.map(([name, family, status, copy, href]) => (
            <a className="method-card" href={href} target="_blank" rel="noreferrer" key={name}>
              <div><Tag tone={status === "Full cache" ? "good" : status === "Pilot cache" ? "warn" : "neutral"}>{status}</Tag><span>{family}</span></div>
              <h3>{name}</h3><p>{copy}</p><ArrowUpRight size={16} />
            </a>
          ))}
        </div>
        <p className="baseline-note">“Integrated” means an upstream implementation and task registry are present; it does not mean the method has a valid full benchmark result. All methods must share the same model, generation budget, dataset subset, parser, and fingerprint before comparison.</p>
      </section>

      <section className="section cases-section" id="cases">
        <div className="section-heading">
          <div>
            <span className="kicker">05 / Clinical cases</span>
            <h2>Rescues, harms, and metric traps.</h2>
          </div>
          <p>
            Aggregate gains are incomplete without the outputs behind them. These examples are drawn directly from cached IU-Xray records.
          </p>
        </div>
        <div className="case-grid">
          <article className="case-card">
            <div className="case-image"><img src="/cases/ce-rescue.png" alt="IU-Xray chest radiograph for a rescue case" /><Tag tone="good">Rescue</Tag></div>
            <div className="case-body">
              <small>IU-Xray · closed-ended</small>
              <h3>Is the cardiomediastinal silhouette abnormal?</h3>
              <div className="answer wrong"><span>Baseline</span><p>Yes.</p><X size={16} /></div>
              <div className="answer correct"><span>Calibrated</span><p>No.</p><Check size={16} /></div>
              <p className="reference">Reference: <b>No.</b> · sequence margin 0.912, below frozen threshold 1.246.</p>
            </div>
          </article>

          <article className="case-card">
            <div className="case-image"><img src="/cases/ce-harm.png" alt="IU-Xray chest radiograph for a harmful flip" /><Tag tone="bad">Harm</Tag></div>
            <div className="case-body">
              <small>IU-Xray · closed-ended</small>
              <h3>Are there degenerative changes of the spine?</h3>
              <div className="answer correct"><span>Baseline</span><p>Yes.</p><Check size={16} /></div>
              <div className="answer wrong"><span>Calibrated</span><p>No.</p><X size={16} /></div>
              <p className="reference">Reference: <b>Yes.</b> · a reminder that the global correction is not clinically uniform.</p>
            </div>
          </article>

          <article className="case-card case-wide">
            <div className="case-image"><img src="/cases/report-normal.png" alt="IU-Xray chest radiograph with normal findings" /><Tag tone="good">Text gain</Tag></div>
            <div className="case-body">
              <small>IU-Xray · report generation</small>
              <h3>When longer phrasing better matches a normal report</h3>
              <div className="report-answer">
                <span>Baseline · ROUGE-L 6.9</span>
                <p>“The chest X-ray appears to be normal, with no significant abnormalities detected.”</p>
              </div>
              <div className="report-answer emphasized">
                <span>Source word center · ROUGE-L 21.7</span>
                <p>“The chest X-ray shows no acute cardiopulmonary pathology… no visible signs of immediate or severe issues affecting the heart and lungs.”</p>
              </div>
              <p className="reference">Reference: lungs clear; no effusion or pneumothorax; heart and mediastinum normal.</p>
            </div>
          </article>

          <article className="case-card case-wide">
            <div className="case-image"><img src="/cases/report-metric-trap.png" alt="IU-Xray chest radiograph illustrating a metric trap" /><Tag tone="warn">Metric trap</Tag></div>
            <div className="case-body">
              <small>IU-Xray · report generation</small>
              <h3>A better overlap score can still omit clinical findings</h3>
              <div className="report-answer emphasized">
                <span>ROUGE-L 4.2 → 21.5</span>
                <p>Selected output: “No acute cardiopulmonary pathology.”</p>
              </div>
              <p className="reference">
                Reference also contains prominent interstitial markings, small bilateral pleural
                effusions, catheter tubing, and degenerative joint disease. Text metrics alone do
                not establish hallucination reduction.
              </p>
            </div>
          </article>
        </div>
      </section>

      <section className="section limits-section" id="limits">
        <div className="section-heading">
          <div>
            <span className="kicker">06 / Claim boundary</span>
            <h2>What the current evidence does—and does not—support.</h2>
          </div>
        </div>
        <div className="claim-grid">
          <article className="claim support">
            <Check size={22} />
            <h3>Supported</h3>
            <ul>
              <li>Sequence-interface calibration materially improves the full MIMIC CE benchmark.</li>
              <li>Source-conditioned report phrasing improves overlap metrics on IU-Xray and MIMIC.</li>
              <li>Visual style interventions expose measurable prediction sensitivity and failure modes.</li>
            </ul>
          </article>
          <article className="claim open">
            <CircleAlert size={22} />
            <h3>Still open</h3>
            <ul>
              <li>A source-guided DG component has not yet shown an independent ≥3pp gain.</li>
              <li>Orbit risk did not pass LODO; locked unknown-hospital testing was not triggered.</li>
              <li>Clinical report gains require paired RadGraph/CheXbert evidence, not ROUGE alone.</li>
            </ul>
          </article>
          <article className="claim reject">
            <X size={22} />
            <h3>Not claimed</h3>
            <ul>
              <li>No guarantee for arbitrary CT, MRI, ultrasound, or fundus shifts.</li>
              <li>No non-vacuous open-ended conformal guarantee from the current pilot.</li>
              <li>No assertion that every appearance normalization is clinically safe.</li>
            </ul>
          </article>
        </div>
      </section>

      <section className="section reproduce">
        <div>
          <span className="kicker">07 / Reproducibility</span>
          <h2>Every number begins with a fingerprint.</h2>
          <p>
            The public package includes immutable reference artifacts, resumable runners,
            dataset policies, and explicit method registries.
          </p>
        </div>
        <div className="code-card">
          <div className="terminal-head"><i /><i /><i /><span>quick start</span></div>
          <pre><code>{`conda env create -f environment.yml\nconda activate anchor\nbash scripts/run_smoke.sh\nbash scripts/summarize_results.sh`}</code></pre>
          <a href={repo} target="_blank" rel="noreferrer">Open repository <ArrowUpRight size={15} /></a>
        </div>
      </section>

      <footer>
        <div className="brand"><span className="brand-mark">A</span><span>ANCHOR</span></div>
        <p>Research artifact · Not for clinical use · Results reflect cached experiments in this repository.</p>
        <a href={repo} target="_blank" rel="noreferrer"><Github size={17} /> dfdu233/ANCHOR</a>
      </footer>
    </main>
  );
}
