"""Pipeline and model-adapter contracts without neural-network inference."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from baseline import detection, ocr, pipeline, structure


class ArrayTensor:
    """Minimal tensor interface used by the YOLO adapter."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return ArrayTensor(self.value[index])

    def item(self):
        return self.value.item()

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def prediction_box(label, coordinates):
    return SimpleNamespace(
        cls=ArrayTensor([label]), xyxy=ArrayTensor([coordinates])
    )


@pytest.mark.parametrize('label', ['Table', 'table'])
def test_detect_filters_clips_and_sorts(monkeypatch, label):
    result = SimpleNamespace(
        names={0: label, 1: 'figure'},
        boxes=[
            prediction_box(0, [10.2, 30.1, 90.1, 110]),
            prediction_box(1, [1, 1, 30, 30]),
            prediction_box(0, [-2, 2, 20, 12]),
            prediction_box(0, [105, 20, 110, 50]),
        ],
    )
    model = Mock()
    model.predict.return_value = [result]
    monkeypatch.setattr(detection, 'table_detector', lambda: model)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)

    assert detection.detect(image) == [(0, 2, 20, 12), (10, 30, 91, 100)]
    args, kwargs = model.predict.call_args
    assert args[0] is image
    assert kwargs == dict(imgsz=1024, conf=0.25, device='cpu', verbose=False)


def test_ocr_preserves_cell_order_and_normalizes(monkeypatch):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, 30:] = 200
    cells = [
        {'bbox': [0, 0, 30, 20], 'text': ''},
        {'bbox': [70, 0, 80, 20], 'text': 'stale'},
        {'bbox': [30, 0, 60, 20], 'text': ''},
    ]
    model = Mock()
    model.predict.return_value = iter([
        {'rec_text': '  Alpha\n beta '}, {'rec_text': '42'},
    ])
    monkeypatch.setattr(ocr, 'ocr_model', lambda: model)

    ocr.recognize(image, {'cells': cells})

    assert [cell['text'] for cell in cells] == ['Alpha beta', '', '42']
    args, kwargs = model.predict.call_args
    crops = args[0]
    assert len(crops) == 2
    assert crops[0].shape == crops[1].shape == (20, 30, 3)
    assert np.all(crops[0] == 0) and np.all(crops[1] == 200)
    assert kwargs == {'batch_size': 8}


def test_ocr_rejects_wrong_result_count(monkeypatch):
    model = Mock()
    model.predict.return_value = iter([])
    monkeypatch.setattr(ocr, 'ocr_model', lambda: model)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match='unexpected result count'):
        ocr.recognize(image, {'cells': [{'bbox': [0, 0, 10, 10]}]})


def test_pipeline_end_to_end(monkeypatch):
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    inverse = np.array([[1, 0, 5], [0, 1, 7]], dtype=float)
    monkeypatch.setattr(pipeline, 'preprocess', lambda image: (image, inverse))
    monkeypatch.setattr(pipeline, 'detect', lambda image: [(10, 20, 50, 60)])
    structure_model = Mock()
    structure_model.predict.return_value = iter([{
        'structure': ['<tr>', '<td></td>', '</tr>'],
        'bbox': [[0, 0, 40, 0, 40, 40, 0, 40]],
    }])
    ocr_model = Mock()
    ocr_model.predict.return_value = iter([{'rec_text': ' 42 '}])
    monkeypatch.setattr(structure, 'structure_model', lambda: structure_model)
    monkeypatch.setattr(ocr, 'ocr_model', lambda: ocr_model)
    detections = []

    result = pipeline.solve(image, detections=detections)

    assert detections == [{'bbox': [15, 27, 55, 67]}]
    assert result == {'tables': [{
        'bbox': [15, 27, 55, 67], 'rows': 1, 'cols': 1,
        'cells': [{'r0': 0, 'r1': 0, 'c0': 0, 'c1': 0,
                   'bbox': [15, 27, 55, 67], 'text': '42'}],
    }]}
    assert structure_model.predict.call_args.args[0].shape == (40, 40, 3)


def test_detection_survives_rejected_structure(monkeypatch):
    image = np.full((30, 30, 3), 255, dtype=np.uint8)
    inverse = np.array([[1, 0, 5], [0, 1, 7]], dtype=float)
    monkeypatch.setattr(pipeline, 'preprocess', lambda image: (image, inverse))
    monkeypatch.setattr(pipeline, 'detect', lambda image: [(1, 2, 10, 12)])
    monkeypatch.setattr(pipeline, 'recover', lambda box, image: None)
    recognize = Mock()
    monkeypatch.setattr(pipeline, 'recognize', recognize)
    detections = []

    assert pipeline.solve(image, detections=detections) == {'tables': []}
    assert detections == [{'bbox': [6, 9, 15, 19]}]
    recognize.assert_not_called()


def test_pipeline_without_detections(monkeypatch):
    monkeypatch.setattr(pipeline, 'detect', lambda image: [])
    recover = Mock()
    monkeypatch.setattr(pipeline, 'recover', recover)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert pipeline.solve(image) == {'tables': []}
    recover.assert_not_called()
