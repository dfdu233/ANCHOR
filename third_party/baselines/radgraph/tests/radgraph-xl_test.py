from radgraph import RadGraph, F1RadGraph
from radgraph import get_radgraph_processed_annotations


def test_1():
    radgraph = RadGraph(model_type="radgraph-xl")
    annotations = radgraph(["no acute cardiopulmonary abnormality"])
    processed_annotations = get_radgraph_processed_annotations(annotations)
    assert processed_annotations == {'processed_annotations': [
        {'observation': 'acute', 'observation_start_ix': [1], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [1]},
        {'observation': 'abnormality', 'observation_start_ix': [3], 'located_at': ['cardiopulmonary'],
         'located_at_start_ix': [[2]], 'tags': ['definitely absent'], 'suggestive_of': None,
         'observation_end_ix': [3]}], 'radgraph_annotations': {'0': {'text': 'no acute cardiopulmonary abnormality',
                                                                     'entities': {'1': {'tokens': 'acute',
                                                                                        'label': 'Observation::definitely absent',
                                                                                        'start_ix': 1, 'end_ix': 1,
                                                                                        'relations': []},
                                                                                  '2': {'tokens': 'cardiopulmonary',
                                                                                        'label': 'Anatomy::definitely present',
                                                                                        'start_ix': 2, 'end_ix': 2,
                                                                                        'relations': []},
                                                                                  '3': {'tokens': 'abnormality',
                                                                                        'label': 'Observation::definitely absent',
                                                                                        'start_ix': 3, 'end_ix': 3,
                                                                                        'relations': [
                                                                                            ['located_at', '2']]}},
                                                                     'data_source': None, 'data_split': 'inference'}},
        'start_ix_to_label': {"1": 'Observation::definitely absent', "2": 'Anatomy::definitely present',
                              "3": 'Observation::definitely absent'},
        'radgraph_text': 'no acute cardiopulmonary abnormality'}


def test_2():
    radgraph = RadGraph(model_type="radgraph-xl")
    annotations = radgraph([
        "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct"])
    processed_annotations = get_radgraph_processed_annotations(annotations)
    assert processed_annotations == {'processed_annotations': [
        {'observation': 'et tube', 'observation_start_ix': [0], 'located_at': ['2 cm above'],
         'located_at_start_ix': [[3]], 'tags': ['definitely present'], 'suggestive_of': None,
         'observation_end_ix': [1]},
        {'observation': 'retraction', 'observation_start_ix': [8], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely present'], 'suggestive_of': None, 'observation_end_ix': [8]},
        {'observation': 'consolidations', 'observation_start_ix': [19], 'located_at': ['bibasilar'],
         'located_at_start_ix': [[18]], 'tags': ['definitely present'], 'suggestive_of': None,
         'observation_end_ix': [19]}], 'radgraph_annotations': {'0': {
        'text': 'et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct',
        'entities': {'1': {'tokens': 'et tube', 'label': 'Observation::definitely present', 'start_ix': 0, 'end_ix': 1,
                           'relations': [['located_at', '2']]},
                     '2': {'tokens': '2 cm above', 'label': 'Anatomy::measurement::definitely present', 'start_ix': 3,
                           'end_ix': 5, 'relations': [['modify', '3']]},
                     '3': {'tokens': 'carina', 'label': 'Anatomy::definitely present', 'start_ix': 7, 'end_ix': 7,
                           'relations': []},
                     '4': {'tokens': 'retraction', 'label': 'Observation::definitely present', 'start_ix': 8,
                           'end_ix': 8, 'relations': []},
                     '5': {'tokens': 'bibasilar', 'label': 'Anatomy::definitely present', 'start_ix': 18, 'end_ix': 18,
                           'relations': []},
                     '6': {'tokens': 'consolidations', 'label': 'Observation::definitely present', 'start_ix': 19,
                           'end_ix': 19, 'relations': [['located_at', '5']]}}, 'data_source': None,
        'data_split': 'inference'}}, 'start_ix_to_label': {"0": 'Observation::definitely present',
                                                           "3": 'Anatomy::measurement::definitely present',
                                                           "7": 'Anatomy::definitely present',
                                                           "8": 'Observation::definitely present',
                                                           "18": 'Anatomy::definitely present',
                                                           "19": 'Observation::definitely present'},
        'radgraph_text': 'et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct'}


def test_3():
    radgraph = RadGraph(model_type="radgraph-xl")
    annotations = radgraph([
        "there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed"])
    processed_annotations = get_radgraph_processed_annotations(annotations)
    assert processed_annotations == {'processed_annotations': [
        {'observation': 'significant change', 'observation_start_ix': [3, 4], 'located_at': [],
         'located_at_start_ix': [], 'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [3, 4]},
        {'observation': 'feeding tube', 'observation_start_ix': [10, 11], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [10, 11]},
        {'observation': 'nasogastric', 'observation_start_ix': [13], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [13]},
        {'observation': 'tube', 'observation_start_ix': [14], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [14]}], 'radgraph_annotations': {
        '0': {
            'text': 'there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed',
            'entities': {
                '1': {'tokens': 'significant', 'label': 'Observation::definitely absent', 'start_ix': 3, 'end_ix': 3,
                      'relations': [['modify', '2']]},
                '2': {'tokens': 'change', 'label': 'Observation::definitely absent', 'start_ix': 4, 'end_ix': 4,
                      'relations': []},
                '3': {'tokens': 'feeding', 'label': 'Observation::definitely absent', 'start_ix': 10, 'end_ix': 10,
                      'relations': [['modify', '4']]},
                '4': {'tokens': 'tube', 'label': 'Observation::definitely absent', 'start_ix': 11, 'end_ix': 11,
                      'relations': []},
                '5': {'tokens': 'nasogastric', 'label': 'Observation::definitely absent', 'start_ix': 13, 'end_ix': 13,
                      'relations': []},
                '6': {'tokens': 'tube', 'label': 'Observation::definitely absent', 'start_ix': 14, 'end_ix': 14,
                      'relations': []}}, 'data_source': None, 'data_split': 'inference'}},
        'start_ix_to_label': {"3": 'Observation::definitely absent',
                              "4": 'Observation::definitely absent',
                              "10": 'Observation::definitely absent',
                              "11": 'Observation::definitely absent',
                              "13": 'Observation::definitely absent',
                              "14": 'Observation::definitely absent'},
        'radgraph_text': 'there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed'}


def test_4():
    radgraph = RadGraph(model_type="radgraph-xl")
    annotations = radgraph(["unchanged mild pulmonary edema no radiographic evidence pneumonia"])
    processed_annotations = get_radgraph_processed_annotations(annotations)
    assert processed_annotations == {'processed_annotations': [
        {'observation': 'mild', 'observation_start_ix': [1], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely present'], 'suggestive_of': None, 'observation_end_ix': [1]},
        {'observation': 'unchanged edema', 'observation_start_ix': [0, 3], 'located_at': ['pulmonary'],
         'located_at_start_ix': [[2]], 'tags': ['definitely present'], 'suggestive_of': None,
         'observation_end_ix': [0, 3]},
        {'observation': 'pneumonia', 'observation_start_ix': [7], 'located_at': [], 'located_at_start_ix': [],
         'tags': ['definitely absent'], 'suggestive_of': None, 'observation_end_ix': [7]}], 'radgraph_annotations': {
        '0': {'text': 'unchanged mild pulmonary edema no radiographic evidence pneumonia', 'entities': {
            '1': {'tokens': 'unchanged', 'label': 'Observation::definitely present', 'start_ix': 0, 'end_ix': 0,
                  'relations': [['modify', '4']]},
            '2': {'tokens': 'mild', 'label': 'Observation::definitely present', 'start_ix': 1, 'end_ix': 1,
                  'relations': []},
            '3': {'tokens': 'pulmonary', 'label': 'Anatomy::definitely present', 'start_ix': 2, 'end_ix': 2,
                  'relations': []},
            '4': {'tokens': 'edema', 'label': 'Observation::definitely present', 'start_ix': 3, 'end_ix': 3,
                  'relations': [['located_at', '3']]},
            '5': {'tokens': 'pneumonia', 'label': 'Observation::definitely absent', 'start_ix': 7, 'end_ix': 7,
                  'relations': []}}, 'data_source': None, 'data_split': 'inference'}},
                                     'start_ix_to_label': {"0": 'Observation::definitely present',
                                                           "1": 'Observation::definitely present',
                                                           "2": 'Anatomy::definitely present',
                                                           "3": 'Observation::definitely present',
                                                           "7": 'Observation::definitely absent'},
                                     'radgraph_text': 'unchanged mild pulmonary edema no radiographic evidence pneumonia'}
