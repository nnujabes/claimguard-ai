import base64
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from backend.ui_pipeline import AnalysisJob, analyze_image, prepare_uploads


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'image.png'
        Image.new('RGB', (32, 24), 'blue').save(self.image)

    def upload(self, name='image.png'):
        return {'name': name, 'data': base64.b64encode(self.image.read_bytes()).decode()}

    def official(self, model, path, **kwargs):
        return {'model': model, 'score': .8, 'elapsed_sec': .1,
                'meta': {'is_official_model': True}, 'artifacts': {}}

    def test_upload_validation_and_safe_paths(self):
        prepared = prepare_uploads([self.upload('../../image.png')], self.root / 'uploads')
        self.assertEqual(prepared[0]['name'], 'image.png')
        self.assertEqual(prepared[0]['path'].parent, self.root / 'uploads')
        self.assertEqual(len(prepare_uploads([self.upload()] * 20, self.root / 'twenty')), 20)
        for files in ([], [self.upload()] * 21, [{'data': 'bad!'}],
                      [{'data': base64.b64encode(b'not an image').decode()}]):
            with self.assertRaises(Exception):
                prepare_uploads(files, self.root / 'invalid')

    def test_real_results_are_fused_after_all_three_models(self):
        with patch('backend.ui_pipeline.run_model', side_effect=self.official) as runner:
            result = analyze_image(self.image, 'photo.png')
        self.assertEqual([call.args[0] for call in runner.call_args_list], ['trufor', 'fused', 'recapture'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['score'], .8)
        self.assertEqual(result['route'], 'HUMAN_REVIEW')
        self.assertTrue(all(call.kwargs['env_overrides']['CG_STRICT_MODE'] == '1' for call in runner.call_args_list))
        self.assertEqual([call.kwargs['timeout'] for call in runner.call_args_list], [180, 180, 180])

    def test_selected_threshold_is_recorded_and_used(self):
        with patch('backend.ui_pipeline.run_model', side_effect=self.official):
            result = analyze_image(self.image, 'photo.png', threshold=.81)
        self.assertEqual(result['threshold'], .81)
        self.assertEqual(result['route'], 'AOS')

    def test_one_failure_does_not_skip_other_models_or_create_a_score(self):
        def run(model, path, **kwargs):
            if model == 'fused':
                raise RuntimeError('GPU failure')
            return self.official(model, path)
        with patch('backend.ui_pipeline.run_model', side_effect=run) as runner:
            result = analyze_image(self.image, 'photo.png')
        self.assertEqual(runner.call_count, 3)
        self.assertIsNone(result['score'])
        self.assertIsNone(result['scores']['fused'])
        self.assertEqual(result['route'], 'INCOMPLETE')
        self.assertEqual(result['models']['recapture']['status'], 'complete')
        self.assertEqual(result['artifacts'], {})

    def test_fallback_is_rejected(self):
        with patch('backend.ui_pipeline.run_model', return_value={'score': .9, 'meta': {'is_official_model': False}}):
            result = analyze_image(self.image, 'photo.png')
        self.assertFalse(result['complete'])
        self.assertTrue(all(value is None for value in result['scores'].values()))

    def test_artifacts_come_from_model_outputs(self):
        def run(model, path, **kwargs):
            result = self.official(model, path)
            if model == 'fused':
                result['artifacts'] = {'mask_path': str(self.image), 'heatmap_overlay_path': str(self.image)}
            return result
        with patch('backend.ui_pipeline.run_model', side_effect=run):
            result = analyze_image(self.image, 'photo.png')
        encoded = result['artifacts']['mask'].split(',')[1]
        self.assertEqual(base64.b64decode(encoded), self.image.read_bytes())

    def test_batch_runs_all_images_and_persists_results(self):
        job = AnalysisJob({'id': 'batch', 'files': [self.upload('one.png'), self.upload('two.png')]})
        job.directory = self.root / 'batch'
        with patch('backend.ui_pipeline.run_model', side_effect=self.official) as runner:
            job.run()
        snapshot = job.snapshot()
        self.assertEqual(runner.call_count, 6)
        self.assertEqual(snapshot['status'], 'complete')
        self.assertEqual([r['name'] for r in snapshot['results']], ['one.png', 'two.png'])
        self.assertTrue((job.directory / 'results.json').is_file())

    def test_cancelled_job_does_not_start_models(self):
        cancelled = threading.Event()
        cancelled.set()
        with patch('backend.ui_pipeline.run_model') as runner:
            with self.assertRaises(InterruptedError):
                analyze_image(self.image, 'photo.png', cancelled=cancelled)
        runner.assert_not_called()


if __name__ == '__main__':
    unittest.main()
