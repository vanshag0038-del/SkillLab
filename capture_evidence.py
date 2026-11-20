"""Capture the real local MLflow UI with headless Edge (isolated profile)."""
import json
import mlflow
from urllib.parse import quote
from playwright.sync_api import sync_playwright
from config import ROOT, OUTPUTS, TRACKING_URI


def main():
    manifest = json.loads((OUTPUTS / 'metrics_summary.json').read_text())
    ids = [v['run_id'] for v in manifest['models'].values()]
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment_id = mlflow.MlflowClient().get_run(ids[0]).info.experiment_id
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1050}, device_scale_factor=1)
        page.goto(TRACKING_URI, wait_until='domcontentloaded')
        page.wait_for_timeout(5000)
        page.screenshot(path=str(ROOT / 'screenshots/debug_home.png'))
        print(page.locator('body').inner_text()[:6000])
        page.goto(TRACKING_URI + '/#/compare-runs?runs=' + quote(json.dumps(ids)) + '&experiments=' + quote(json.dumps([experiment_id])), wait_until='domcontentloaded')
        page.wait_for_timeout(5000)
        page.set_viewport_size({'width': 1800, 'height': 2300})
        page.screenshot(path=str(ROOT / 'screenshots/mlflow_compare.png'), full_page=True)
        print('COMPARE:', page.locator('body').inner_text()[:6000])
        page.set_viewport_size({'width': 1600, 'height': 1050})
        page.goto(TRACKING_URI + '/#/models/sentiment-model', wait_until='domcontentloaded')
        page.wait_for_timeout(5000)
        page.screenshot(path=str(ROOT / 'screenshots/registry_aliases.png'), full_page=True)
        print('REGISTRY:', page.locator('body').inner_text()[:6000])
        # Browser rendering of the verbatim terminal transcript, clearly labeled.
        # This is supporting evidence, not a claim to be a native terminal capture.
        import html
        transcript = (OUTPUTS / 'prediction_terminal.txt').read_text(encoding='utf-8')
        page.set_content('<html><body style="background:#101820;color:#e5eef4;padding:28px;font:16px Consolas,monospace"><h2>predict_from_mlflow.py — verbatim execution transcript</h2><pre style="white-space:pre-wrap">' + html.escape(transcript) + '</pre></body></html>')
        page.screenshot(path=str(ROOT / 'screenshots/prediction_terminal.png'), full_page=True)
        browser.close()


if __name__ == '__main__':
    main()
