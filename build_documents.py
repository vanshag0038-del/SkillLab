"""Build the final Word records from the saved, measured project evidence.

Install requirements-evidence.txt first. PDF export and visual QA are separate.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
STUDENT = json.loads((ROOT / 'student.json').read_text(encoding='utf-8'))
RESULT = json.loads((OUT / 'metrics_summary.json').read_text(encoding='utf-8'))
QUIZZES = json.loads((ROOT / 'quiz_answers.json').read_text(encoding='utf-8'))
REPO = 'https://github.com/harsha31bs07-blip/1MS25CI051-mlflow-champion-challenger'
TOPICS = [
    'Introduction to MLOps', 'Introduction to MLflow', 'Tracking', 'Logging Functions',
    'Multiple Experiments and Runs', 'Autologging', 'Tracking Server', 'Models',
    'Custom Models', 'Model Evaluation', 'Model Registry', 'Projects',
    'MLflow Client', 'MLflow CLI',
]
OBSERVATIONS = [
    'Six baseline and delivery splits record MAE and train/test counts. At test_size 0.2 the five delivery observations give four training rows and one test row; R2 is undefined for that single test row. A CSV has no worksheets: execution, dataset, test_size, seed, id and split identify membership in one CSV. The accompanying XLSX has six separate sheets.',
    'Both baseline and delivery runs were executed in a legacy file store. Experiment meta.yaml describes the experiment; each run has metadata, parameter, metric, tag and artifact files. The complete directory listing is saved. Later server exercises use SQLite for metadata and a separate artifact store.',
    'The baseline and delivery runs share an HTTP tracking server. An experiment groups runs; a run records one execution. Parameters describe settings, metrics measure numeric outcomes, tags hold metadata, artifacts hold files, and a logged model is a reloadable package. One server can serve several clients.',
    'The exact twenty-patient table from the brief is split into sixteen training and four test observations. Logged outputs contain actual stay, predicted stay, signed prediction-minus-actual error and absolute error. This regression exercise illustrates logging and is not a clinical predictor.',
    'Nine runs combine seeds 7, 21 and 42 with test sizes 0.2, 0.3 and 0.4. Compare sizes within a fixed seed. More training rows do not guarantee a lower held-out error because test membership also changes. The saved comparison records counts, MAE and R2 for every run.',
    'Automatic and manual runs use the same linear regression configuration and experiment. Autologging captures estimator parameters, training metrics and a model. Manual logging explicitly captures split settings, counts, held-out metrics and the model. The JSON comparison records what each run actually logged.',
    'The two independent scripts use http://127.0.0.1:5000 and Lab07-Shared-Server, with and without an intercept. They do not need separate servers and can log to the same experiment. A shared tracking server gives a team common provenance, comparisons and artifact access.',
    'The fitted sklearn model was logged and its MLflow URI saved. Loading that package through mlflow.pyfunc reproduces the recorded predictions for x values 2, 6 and 12. The standalone load_and_predict.py entry point reloads the stored model without retraining.',
    'SentimentModel is a custom mlflow.pyfunc.PythonModel. Its logged artifacts bundle Hugging Face weights and tokenizer. load_context restores the pipeline internally; the caller predicts through MLflow. The URI and predictions for two new sentences are recorded.',
    'Linear Regression Experiment logs parameters and held-out MAE. Linear Regression Metrics Experiment also logs held-out MSE and R2. MAE and MSE are lower-is-better; R2 is higher-is-better and can be negative. Neither experiment evaluates on training rows.',
    'Intercept and no-intercept candidates are registered under lab-regression. The lower-MAE version receives the champion alias; the other receives challenger. Reloading models:/lab-regression@champion resolves the model without a fixed version in the inference code.',
    'The source brief repeats LAB 11 for Projects; this record numbers it Lab 12. MLproject declares test_size and random_state. The same train.py was executed at sizes 0.25 and 0.40 with seed 7. MLflow Projects creates the enclosing run, which the entry point resumes.',
    'MlflowClient searches the shared Lab05-Comparison experiment for FINISHED runs, orders metrics.mae ascending and returns the first three. top_three.csv preserves the ranked run IDs and results. Using the same server avoids copying a live database between lab folders.',
    'The CLI version, help and experiment search commands were executed against the tracking store. The transcript preserves their actual output. CLI subprocesses explicitly inherit the configured tracking URI, including when no environment variable was supplied by the caller.',
]


def text(relative):
    return (ROOT / relative).read_text(encoding='utf-8')


def new_document(title):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    section.top_margin = section.bottom_margin = Inches(.62)
    section.left_margin = section.right_margin = Inches(.7)
    normal = doc.styles['Normal']
    normal.font.name, normal.font.size = 'Calibri', Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.05
    for name, size in [('Title', 25), ('Subtitle', 13), ('Heading 1', 17), ('Heading 2', 12)]:
        style = doc.styles[name]
        style.font.name, style.font.size = 'Calibri', Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
    doc.core_properties.title, doc.core_properties.author = title, STUDENT['name']
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    run = footer.add_run(f"{STUDENT['usn']} | ")
    run.font.size = Pt(8)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    return doc


def p(doc, value, style=None):
    return doc.add_paragraph(str(value), style)


def heading(doc, value, level=1, new_page=False):
    if new_page:
        doc.add_page_break()
    doc.add_heading(value, level)


def code(doc, value, max_lines=None):
    lines = value.strip('\n').splitlines()
    if max_lines:
        lines = lines[:max_lines]
    for line in lines:
        para = p(doc, line.expandtabs(4))
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.line_spacing = 1
        for run in para.runs:
            run.font.name, run.font.size = 'Consolas', Pt(7.7)
    p(doc, '')


def table(doc, headers, rows, widths=None):
    result = doc.add_table(rows=1, cols=len(headers))
    result.style = 'Table Grid'
    result.autofit = False
    if widths:
        for col, width in zip(result.columns, widths):
            col.width = Inches(width)
    for cell, value in zip(result.rows[0].cells, headers):
        cell.text = str(value)
        shade = OxmlElement('w:shd')
        shade.set(qn('w:fill'), 'E7EEF4')
        cell._tc.get_or_add_tcPr().append(shade)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    repeat = OxmlElement('w:tblHeader')
    result.rows[0]._tr.get_or_add_trPr().append(repeat)
    for values in rows:
        for cell, value in zip(result.add_row().cells, values):
            cell.text = str(value)
    for row in result.rows:
        if widths:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
        no_split = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(no_split)
        for cell in row.cells:
            cell.vertical_alignment = 1
            for para in cell.paragraphs:
                para.paragraph_format.space_after = Pt(4)
                para.paragraph_format.space_before = Pt(4)
                for run in para.runs:
                    run.font.size = Pt(9)
    p(doc, '')
    return result


def course_value(key):
    value = STUDENT.get(key, '')
    return '____________________________' if not value or 'pending' in value.casefold() else value


def add_quiz(doc, day, separate=False):
    heading(doc, f'Day {day} Quiz Answers', new_page=not separate)
    p(doc, f"{STUDENT['name']} | {STUDENT['usn']} | Section {STUDENT['section']}")
    p(doc, 'Responses and explanations checked against the supplied answer key. Instructor assessment is recorded separately.')
    for number, answer, explanation in QUIZZES[f'day_{day}']:
        p(doc, f'{number}. {answer} - {explanation}')


def lab_code(number):
    source = text('run_labs.py')
    token = '    if number == 1:' if number == 1 else f'    elif number == {number}:'
    start = source.index(token)
    tail = source[start:].splitlines()
    lines = []
    for line in tail[1:]:
        if line.startswith(('    elif ', '    else:')):
            break
        lines.append(line[8:] if line.startswith('        ') else line)
    # Keep literal code excerpts short; full source is included in the package.
    return '\n'.join(lines[:11])


def main():
    doc = new_document('MLflow Skill Lab Completed Record')
    p(doc, 'MLflow Skill Lab Completed Record', 'Title')
    p(doc, 'Labs 01 through 14 and Champion Challenger Mini Project', 'Subtitle')
    p(doc, f"Student name: {STUDENT['name']}")
    p(doc, f"USN: {STUDENT['usn']}    Section: {STUDENT['section']}")
    p(doc, f"Course title: {course_value('course_title')}")
    p(doc, f"Course code: {course_value('course_code')}")
    p(doc, f'Final review: {date.today().strftime("%d %B %Y")} | Measured experiments: 10 November 2026')
    p(doc, REPO)
    p(doc, 'This record presents the completed labs, measured model comparison, registry inference and quiz responses. DistilBERT remains Champion under the rule declared before evaluation. Both saved models reproduce their predictions, and the audit verifies all fourteen lab exercises.')
    heading(doc, 'Assessment', 2)
    table(doc, ['Component', 'Maximum marks', 'Marks awarded'], [
        ['Lab exercises 01 to 14', 30, ''], ['Mini project', 50, ''],
        ['Quiz 1', 10, ''], ['Quiz 2', 10, ''], ['Total', 100, ''],
    ], [3.8, 1.3, 1.6])
    p(doc, 'Instructor signature: ____________________    Date: ______________')
    p(doc, 'AI declaration: OpenAI Codex assisted with requirements, data authoring, code, execution and explanations. The student must review and explain the implementation in the viva.')

    heading(doc, 'Mini Project Method and Results', new_page=True)
    p(doc, 'Task: binary English sentiment classification on 60 fixed original sentences, balanced as 30 POSITIVE and 30 NEGATIVE. Both models receive the same inputs in the same order. Eight separate new sentences are reserved for inference.')
    for role in ['A', 'B']:
        p(doc, f"Model {role}: {RESULT['models'][role]['hf_model_id']}")
    p(doc, 'Promote B only when its macro-F1 exceeds A by at least 0.01 and its warm average latency is at most 5 times A. Otherwise retain A. Higher classification scores are better; lower latency and parameter count are better. The margin avoids replacement for a negligible change and is not a significance test.')
    p(doc, 'CPU inference uses seed 42, four torch threads, batch size 8 and maximum length 128. Model load and one warm-up call are excluded from timing. Exact Hub revisions and library versions are preserved in the metric manifest.')
    a, b = [RESULT['models'][r]['metrics'] for r in ['A', 'B']]
    table(doc, ['Metric', 'A', 'B', 'B minus A'], [[key, f'{a[key]:.6f}', f'{b[key]:.6f}', f'{b[key]-a[key]:+.6f}'] for key in a], [2.7, 1.3, 1.3, 1.4])
    heading(doc, 'Decision', 2)
    p(doc, text('outputs/results.md').split('\n\n', 1)[1])

    heading(doc, 'Evaluation Plots', new_page=True)
    doc.add_picture(str(OUT / 'comparison.png'), width=Inches(6.8))
    p(doc, 'Measured comparison across every recorded metric. Values and units are defined in the results table.')
    for role in ['A', 'B']:
        doc.add_picture(str(OUT / role / 'confusion_matrix.png'), width=Inches(3.5))
        p(doc, f'Model {role} confusion matrix on the same sixty sentences.')

    heading(doc, 'Qualitative Review and New Predictions', new_page=True)
    p(doc, 'The models disagree on twelve inputs. Six specific cases below interpret the observed predictions; the complete prediction files retain every input and reference.')
    comments = pd.read_csv(OUT / 'qualitative_analysis.csv')
    table(doc, ['Input ID', 'Comment'], comments.values.tolist(), [.65, 6.05])
    registry = json.loads(text('outputs/registry.json'))
    p(doc, f"Registry: {registry['name']} | Champion A version {registry['versions']['A']} | Challenger B version {registry['versions']['B']}")
    p(doc, 'predict_from_mlflow.py loads models:/sentiment-model@champion and reads the version and Hub ID from registry metadata. It imports no Hugging Face pipeline. The pyfunc wrapper uses its bundled weights and tokenizer internally.')
    predictions = pd.read_csv(OUT / 'new_data_predictions.csv')
    table(doc, ['ID', 'New input', 'Prediction'], predictions[['id', 'text', 'prediction']].values.tolist(), [.45, 5.2, 1.05])

    heading(doc, 'MLflow Evidence', new_page=True)
    doc.add_picture(str(ROOT / 'screenshots/mlflow_compare.png'), width=Inches(6.1))
    p(doc, 'Actual MLflow compare view of the evaluated candidates. The full resolution image is included under screenshots/.')
    heading(doc, 'Registry Evidence', new_page=True)
    doc.add_picture(str(ROOT / 'screenshots/registry_aliases.png'), width=Inches(6.6))
    p(doc, 'Actual local MLflow registry view showing champion and challenger aliases.')
    heading(doc, 'Prediction Evidence', 2)
    p(doc, 'outputs/prediction_terminal.txt preserves the executed command output. screenshots/prediction_terminal.png is a labeled rendering of that transcript. A native terminal-window capture must be added before final submission; desktop screenshot support is unavailable in this session.')

    for number, topic in enumerate(TOPICS, 1):
        heading(doc, f'Lab {number:02} {topic}', new_page=True)
        p(doc, f'Command: .venv\\Scripts\\python.exe run_labs.py --lab {number}')
        p(doc, f'Full implementation: run_labs.py, branch number == {number}; named entry points are under labs/lab_{number:02}/.')
        heading(doc, 'Code excerpt', 2)
        if number == 12:
            code(doc, text('labs/lab_12/MLproject'))
        else:
            code(doc, lab_code(number))
        heading(doc, 'Observation and answer', 2)
        p(doc, OBSERVATIONS[number-1])
        heading(doc, 'Measured output excerpt', 2)
        relative = f'outputs/labs/lab_{number:02}/terminal.txt'
        lines = [line for line in text(relative).splitlines() if 'View run' not in line and 'View experiment' not in line]
        # Long store listings and CLI help are available in full in the package.
        excerpt = [''.join(c for c in line if ord(c) < 127)[:108] for line in lines[:18]]
        code(doc, '\n'.join(excerpt))
        p(doc, f'Full transcript and supporting tables: outputs/labs/lab_{number:02}/. Code and output excerpts are shortened for page readability.')

    for day in [1, 2]:
        add_quiz(doc, day)
    heading(doc, 'Reproduction and Limitations', new_page=True)
    p(doc, 'Follow README.md from a fresh extraction: create the virtual environment, install constrained dependencies, start the local server, evaluate, register aliases, predict, then run the labs. The ZIP excludes tracking stores and model weights as required by the assignment. Existing run IDs describe the original execution; regenerate outputs and registry versions on a new machine before running live verification.')
    p(doc, 'The original teaching starter scripts were not supplied. The general baseline is deterministic synthetic regression data; delivery and hospital tables reproduce the brief. The repeated Projects heading is numbered Lab 12. Full executable source and all saved transcripts are included.')
    p(doc, 'The sentiment sample is small and not randomly drawn from a target population. AI-authored intended labels were not independently annotated. Training overlap cannot be excluded; CPU timings vary with load. Scores do not establish population performance or statistical significance. Confidence scores are not calibrated probabilities.')
    p(doc, QUIZZES['declaration'])
    p(doc, 'Local tests and model/lab replay passed. SUBMISSION_GUIDE.md lists the final files and remaining administrative steps. Instructor access to the private repository, a native terminal screenshot, course details and LMS submission still require completion. No marks, signature or submission receipt are claimed.')
    p(doc, 'References: supplied assignment and lab book; MLflow PythonModel documentation at https://mlflow.org/docs/latest/ml/model/python_model/; MLflow registry documentation at https://mlflow.org/docs/latest/ml/model-registry/; the two Hugging Face model cards identified above.')
    doc.save(ROOT / 'Completed_Lab_Record.docx')

    quiz = new_document('Completed MLOps Quiz Answers')
    p(quiz, 'Completed MLOps Quiz Answers', 'Title')
    add_quiz(quiz, 1, separate=True)
    add_quiz(quiz, 2)
    quiz.save(ROOT / 'Completed_Quiz_Answers.docx')
    print('Built Completed_Lab_Record.docx and Completed_Quiz_Answers.docx')


if __name__ == '__main__':
    main()
