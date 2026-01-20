
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import os
import base64
import io
from PIL import Image
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from datetime import datetime
import json
import uuid
import glob


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Parcel Damage Detection - Custom Test Dashboard'


def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        session = ort.InferenceSession(model_path)
        print(f"Successfully loaded model: {model_path}")
        return session
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None




def predict(model, img_path):
    try:
        
        image = Image.open(img_path).convert('RGB')
        
        
        image = image.resize((640, 640))
        
        
        img_array = np.array(image).astype(np.float32) / 255.0
        
        
        img_array = img_array.transpose(2, 0, 1)
        
        
        img_array = np.expand_dims(img_array, axis=0)
        
        
        input_name = model.get_inputs()[0].name
        
        
        outputs = model.run(None, {input_name: img_array})
        output = outputs[0]
        
        
        filename = os.path.basename(img_path)
        print(f"Raw output for {filename}: {output[0]}")
        
        
        actual_class = "damaged" if "damaged" in img_path else "undamaged"
        
        
        if output.shape[1] == 2:
            class_0_conf = float(output[0][0])  
            class_1_conf = float(output[0][1])  
            
            print(f"  Class 0 confidence: {class_0_conf:.6f}")
            print(f"  Class 1 confidence: {class_1_conf:.6f}")
            print(f"  Actual class: {actual_class}")
            
            
            
            
            
            
            
            
            
            
            if class_1_conf > class_0_conf:
                pred_class_v1 = "damaged"
                confidence_v1 = class_1_conf
            else:
                pred_class_v1 = "undamaged"
                confidence_v1 = class_0_conf
            
            
            if class_0_conf > class_1_conf:
                pred_class_v2 = "damaged"
                confidence_v2 = class_0_conf
            else:
                pred_class_v2 = "undamaged"
                confidence_v2 = class_1_conf
            
            
            
            threshold = 0.5
            if class_1_conf > threshold:
                pred_class_v3 = "damaged"
                confidence_v3 = class_1_conf
            else:
                pred_class_v3 = "undamaged"
                confidence_v3 = 1.0 - class_1_conf
            
            print(f"  Option 1 (std): {pred_class_v1} (conf: {confidence_v1:.4f})")
            print(f"  Option 2 (inv): {pred_class_v2} (conf: {confidence_v2:.4f})")
            print(f"  Option 3 (thr): {pred_class_v3} (conf: {confidence_v3:.4f})")
            
            
            
            
            
            
            
            
            pred_class, display_confidence = pred_class_v2, confidence_v2  
            
            
            
            print(f"  Final prediction: {pred_class} (confidence: {display_confidence:.4f})")
            print(f"  Correct? {pred_class == actual_class}")
            print("-" * 50)
            
        else:
            
            confidence = float(output.flatten()[0])
            pred_class = "damaged" if confidence > 0.5 else "undamaged"
            display_confidence = confidence if pred_class == "damaged" else 1.0 - confidence
        
        return pred_class, display_confidence
    except Exception as e:
        print(f"Error in prediction for {img_path}: {e}")
        return "Error", 0.0


def predict_alternative(model, img_path):
    """Alternative prediction function if the main one doesn't work"""
    try:
        
        image = Image.open(img_path).convert('RGB')
        image = image.resize((640, 640))
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: img_array})
        output = outputs[0]
        
        
        if output.shape[1] == 2:
            
            
            damaged_score = float(output[0][0])
            undamaged_score = float(output[0][1])
            
            
            
            if damaged_score < undamaged_score:
                
                pred_class = "undamaged"
                confidence = undamaged_score
            else:
                pred_class = "damaged" 
                confidence = damaged_score
                
            return pred_class, confidence
        
        return "Error", 0.0
    except Exception as e:
        print(f"Error in alternative prediction: {e}")
        return "Error", 0.0
    
def analyze_test_directory(test_dir):
    """Analyze the test directory and return information about its structure"""
    if not os.path.exists(test_dir):
        return None, f"Directory does not exist: {test_dir}"
    
    
    damaged_dir = os.path.join(test_dir, 'damaged')
    undamaged_dir = os.path.join(test_dir, 'undamaged')
    
    if not os.path.exists(damaged_dir):
        return None, f"'damaged' folder not found in: {test_dir}"
    
    if not os.path.exists(undamaged_dir):
        return None, f"'undamaged' folder not found in: {test_dir}"
    
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
    
    damaged_files = []
    undamaged_files = []
    
    for ext in image_extensions:
        damaged_files.extend(glob.glob(os.path.join(damaged_dir, f'*{ext}')))
        undamaged_files.extend(glob.glob(os.path.join(undamaged_dir, f'*{ext}')))
    
    if len(damaged_files) == 0:
        return None, f"No image files found in damaged folder: {damaged_dir}"
    
    if len(undamaged_files) == 0:
        return None, f"No image files found in undamaged folder: {undamaged_dir}"
    
    info = {
        'damaged_dir': damaged_dir,
        'undamaged_dir': undamaged_dir,
        'damaged_files': damaged_files,
        'undamaged_files': undamaged_files,
        'damaged_count': len(damaged_files),
        'undamaged_count': len(undamaged_files),
        'total_count': len(damaged_files) + len(undamaged_files)
    }
    
    return info, f"Found {len(damaged_files)} damaged and {len(undamaged_files)} undamaged images"


def evaluate_model_on_custom_set(model_path, test_dir):
    """Evaluate a model on a custom test directory"""
    
    
    dir_info, message = analyze_test_directory(test_dir)
    if dir_info is None:
        return None, message
    
    print(f"Directory analysis: {message}")
    
    
    model = load_model(model_path)
    if model is None:
        return None, f"Failed to load model: {model_path}"
    
    
    y_true = []
    y_pred = []
    confidences = []
    prediction_details = []
    
    print(f"Processing {dir_info['damaged_count']} damaged images...")
    
    
    for i, img_path in enumerate(dir_info['damaged_files']):
        if i % 10 == 0:
            print(f"  Processing damaged image {i+1}/{dir_info['damaged_count']}")
        
        pred_class, confidence = predict(model, img_path)
        
        y_true.append(1)  
        y_pred.append(1 if pred_class == 'damaged' else 0)
        
        
        
        roc_score = 1.0 - confidence if pred_class == 'undamaged' else confidence
        confidences.append(roc_score)
        
        prediction_details.append({
            'file': os.path.basename(img_path),
            'true_class': 'damaged',
            'pred_class': pred_class,
            'confidence': confidence,
            'correct': pred_class == 'damaged'
        })
    
    print(f"Processing {dir_info['undamaged_count']} undamaged images...")
    
    
    for i, img_path in enumerate(dir_info['undamaged_files']):
        if i % 10 == 0:
            print(f"  Processing undamaged image {i+1}/{dir_info['undamaged_count']}")
        
        pred_class, confidence = predict(model, img_path)
        
        y_true.append(0)  
        y_pred.append(1 if pred_class == 'damaged' else 0)
        
        
        roc_score = confidence if pred_class == 'damaged' else 1.0 - confidence
        confidences.append(roc_score)
        
        prediction_details.append({
            'file': os.path.basename(img_path),
            'true_class': 'undamaged',
            'pred_class': pred_class,
            'confidence': confidence,
            'correct': pred_class == 'undamaged'
        })
    
    
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        
        tp = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
        tn = sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0])
        fp = sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
        fn = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    
    fpr, tpr, _ = roc_curve(y_true, confidences)
    roc_auc = auc(fpr, tpr)
    
    
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, confidences)
    
    print(f"\nResults Summary:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  True Positives: {tp}, True Negatives: {tn}")
    print(f"  False Positives: {fp}, False Negatives: {fn}")
    
    results = {
        'test_directory': test_dir,
        'model_path': model_path,
        'directory_info': dir_info,
        'confusion_matrix': cm.tolist(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc),
        'roc_curve': {
            'fpr': [float(x) for x in fpr],
            'tpr': [float(x) for x in tpr],
        },
        'pr_curve': {
            'precision': [float(x) for x in pr_precision],
            'recall': [float(x) for x in pr_recall],
        },
        'prediction_details': prediction_details,
        'counts': {
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    }
    
    return results, "Evaluation completed successfully"

def create_metric_card(title, value, subtitle=None, color="primary"):
    if isinstance(value, float) and 0 <= value <= 1:
        display_value = f"{value*100:.2f}%"
    else:
        display_value = str(value)
        
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted"),
            html.H4(display_value, className=f"text-{color}"),
            html.P(subtitle, className="card-text text-muted small") if subtitle else None,
        ]),
        className="shadow-sm mb-4"
    )


def create_comparison_card(title, value1, value2, model1_name="Model 1", model2_name="Model 2", higher_is_better=True):
    if isinstance(value1, float) and 0 <= value1 <= 1:
        value1_display = value1 * 100
        value2_display = value2 * 100
        diff = value1_display - value2_display
        format_str = "%.2f%%"
    else:
        value1_display = value1
        value2_display = value2
        diff = value1 - value2
        format_str = "%.4f"
        
    is_better = diff > 0 if higher_is_better else diff < 0
    color = "success" if is_better else "danger"
    icon = "↑" if diff > 0 else "↓"
    
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-subtitle text-muted"),
            html.Div([
                html.Div([
                    html.Small(f"{model1_name}:", className="text-muted"),
                    html.H5(format_str % value1_display, className=f"text-primary mb-1")
                ]),
                html.Div([
                    html.Small(f"{model2_name}:", className="text-muted"),
                    html.H5(format_str % value2_display, className="text-secondary mb-1")
                ]),
                html.Hr(className="my-2"),
                html.Div([
                    html.Small("Difference:", className="text-muted"),
                    html.Span(f" {icon} {abs(diff):.2f}%", className=f"text-{color} fw-bold")
                ])
            ]),
        ]),
        className="shadow-sm mb-4"
    )


def create_model_path_modal():
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add Second Model Path")),
            dbc.ModalBody([
                html.P("Please enter the path to your second model for comparison:"),
                dbc.Input(
                    id="modal-model2-path",
                    type="text",
                    placeholder="Enter path to second model (e.g., models/best0.onnx)",
                    value="models/best0.onnx"
                ),
                dbc.FormText("This will be used as the baseline/previous model for comparison."),
                html.Hr(),
                html.P("Current Configuration:", className="fw-bold"),
                html.Div(id="modal-current-config", className="text-muted small")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="modal-cancel", className="ms-auto", n_clicks=0),
                dbc.Button("Save & Compare", id="modal-save", color="primary", n_clicks=0),
            ]),
        ],
        id="model-path-modal",
        is_open=False,
        size="lg"
    )

app.layout = dbc.Container([
    
    create_model_path_modal(),
    
    
    dbc.Row([
        dbc.Col([
            html.H1("Custom Test Set Evaluation Dashboard", className="text-center my-4"),
            html.P("Evaluate your ONNX models on custom test directories", 
                  className="text-center text-muted mb-4"),
        ])
    ]),
    
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Configuration"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Evaluation Mode:", className="fw-bold"),
                            dbc.RadioItems(
                                options=[
                                    {"label": "Single Model", "value": "single"},
                                    {"label": "Compare Two Models", "value": "compare"}
                                ],
                                value="single",
                                id="mode-selector",
                                inline=True,
                                className="mb-3"
                            ),
                        ], width=12)
                    ]),
                    
                    
                    html.Div(id="single-model-inputs", children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Model Path:", className="fw-bold"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="model-path-input",
                                        type="text",
                                        placeholder="Enter path to your .onnx model file",
                                        value="models/best1.onnx"
                                    ),
                                    dbc.Button("Browse", id="browse-model1", outline=True, color="secondary")
                                ]),
                                dbc.FormText("Example: models/best1.onnx or /path/to/your/model.onnx")
                            ], width=6),
                            dbc.Col([
                                html.Label("Test Directory:", className="fw-bold"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="test-dir-input",
                                        type="text",
                                        placeholder="Enter path to test directory",
                                        value="test/testing_on_real/val"
                                    ),
                                    dbc.Button("Browse", id="browse-test-dir", outline=True, color="secondary")
                                ]),
                                dbc.FormText("Directory should contain 'damaged' and 'undamaged' folders")
                            ], width=6)
                        ]),
                    ]),
                    
                    
                    html.Div(id="comparison-model-inputs", style={"display": "none"}, children=[
                        dbc.Row([
                            dbc.Col([
                                html.Label("Model 1 Path (Current):", className="fw-bold"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="model1-path-input",
                                        type="text",
                                        placeholder="Enter path to first model",
                                        value="models/best1.onnx"
                                    ),
                                    dbc.Button("Browse", id="browse-model1-comp", outline=True, color="secondary")
                                ]),
                                dbc.FormText("Current/newer model")
                            ], width=4),
                            dbc.Col([
                                html.Label("Model 2 Path (Baseline):", className="fw-bold"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="model2-path-input",
                                        type="text",
                                        placeholder="Enter path to second model",
                                        value="models/best0.onnx"
                                    ),
                                    dbc.Button("Add Path", id="add-model2-path", color="primary", outline=True)
                                ]),
                                dbc.FormText("Previous/baseline model")
                            ], width=4),
                            dbc.Col([
                                html.Label("Test Directory:", className="fw-bold"),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="test-dir-compare-input",
                                        type="text",
                                        placeholder="Enter path to test directory",
                                        value="test/testing_on_real/val"
                                    ),
                                    dbc.Button("Browse", id="browse-test-dir-comp", outline=True, color="secondary")
                                ]),
                                dbc.FormText("Directory should contain 'damaged' and 'undamaged' folders")
                            ], width=4)
                        ]),
                    ]),
                    
                    html.Hr(),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Analyze Directory", id="analyze-button", color="info", className="me-2"),
                            dbc.Button("Run Evaluation", id="evaluate-button", color="primary", disabled=True),
                            dbc.Button("Quick Compare", id="quick-compare-button", color="success", className="ms-2", style={"display": "none"}),
                        ], width=12)
                    ]),
                    
                    html.Div(id="directory-analysis", className="mt-3")
                ])
            ], className="shadow-sm mb-4")
        ])
    ]),
    
    
    html.Div(id="results-section", style={"display": "none"}, children=[
        
        html.Div(id="single-results", children=[
            
            dbc.Row([
                dbc.Col([html.Div(id="accuracy-card")], width=3),
                dbc.Col([html.Div(id="precision-card")], width=3),
                dbc.Col([html.Div(id="recall-card")], width=3),
                dbc.Col([html.Div(id="f1-card")], width=3),
            ]),
            
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confusion Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="confusion-matrix-graph", config={'displayModeBar': False})
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ROC Curve"),
                        dbc.CardBody([
                            dcc.Graph(id="roc-curve-graph", config={'displayModeBar': False})
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6)
            ]),
            
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Precision-Recall Curve"),
                        dbc.CardBody([
                            dcc.Graph(id="pr-curve-graph", config={'displayModeBar': False})
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            "Sample Predictions",
                            dbc.Button("Enable Comparison Mode", id="enable-comparison", 
                                     color="outline-primary", size="sm", className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.Div(id="predictions-table")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6)
            ])
        ]),
        
        html.Div(id="comparison-results", style={"display": "none"}, children=[
            
            dbc.Row([
                dbc.Col([html.Div(id="accuracy-comparison-card")], width=3),
                dbc.Col([html.Div(id="precision-comparison-card")], width=3),
                dbc.Col([html.Div(id="recall-comparison-card")], width=3),
                dbc.Col([html.Div(id="f1-comparison-card")], width=3),
            ]),
            
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Confusion Matrices Comparison"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Model 1 (Current)", className="text-center mb-3", id="cm1-title"),
                                    dcc.Graph(id="cm1-graph", config={'displayModeBar': False})
                                ], width=6),
                                dbc.Col([
                                    html.H6("Model 2 (Baseline)", className="text-center mb-3", id="cm2-title"),
                                    dcc.Graph(id="cm2-graph", config={'displayModeBar': False})
                                ], width=6)
                            ])
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ]),
            
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("ROC Curves Comparison"),
                        dbc.CardBody([
                            dcc.Graph(id="roc-comparison-graph", config={'displayModeBar': False})
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Precision-Recall Curves Comparison"),
                        dbc.CardBody([
                            dcc.Graph(id="pr-comparison-graph", config={'displayModeBar': False})
                        ])
                    ], className="shadow-sm mb-4")
                ], width=6)
            ]),
            
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            "Detailed Metrics Comparison",
                            dbc.Badge("Model Comparison Active", color="success", className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.Div(id="detailed-comparison-table")
                        ])
                    ], className="shadow-sm mb-4")
                ], width=12)
            ])
        ])
    ]),
    
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Additional Tools"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Export Results", id="export-results", color="outline-primary", className="me-2"),
                            dbc.Button("Save Configuration", id="save-config", color="outline-secondary", className="me-2"),
                            dbc.Button("Load Configuration", id="load-config", color="outline-info"),
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.P("Model Paths:", className="mb-1 fw-bold"),
                                html.Small(id="current-models-display", className="text-muted")
                            ])
                        ], width=6)
                    ])
                ])
            ], className="shadow-sm mb-4")
        ])
    ]),
    
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Custom Test Set Evaluation Dashboard • Enhanced Model Comparison • " + 
                  datetime.now().strftime("%Y-%m-%d %H:%M"),
                  className="text-center text-muted small")
        ])
    ])
], fluid=True, className="pb-5")

@app.callback(
    [Output("directory-analysis", "children"),
     Output("evaluate-button", "disabled")],
    Input("analyze-button", "n_clicks"),
    [State("mode-selector", "value"),
     State("test-dir-input", "value"),
     State("test-dir-compare-input", "value")],
    prevent_initial_call=True
)
def analyze_directory(n_clicks, mode, test_dir_single, test_dir_compare):
    test_dir = test_dir_compare if mode == "compare" else test_dir_single
    
    if not test_dir or test_dir.strip() == "":
        return dbc.Alert("Please enter a test directory path", color="warning"), True
    
    dir_info, message = analyze_test_directory(test_dir.strip())
    
    if dir_info is None:
        return dbc.Alert(f"Error: {message}", color="danger"), True
    else:
        analysis_content = dbc.Alert([
            html.H6("✅ Directory Analysis Successful", className="alert-heading"),
            html.P(message),
            html.Hr(),
            html.P([
                f"📁 Test Directory: {test_dir}", html.Br(),
                f"🔴 Damaged Images: {dir_info['damaged_count']}", html.Br(),
                f"🟢 Undamaged Images: {dir_info['undamaged_count']}", html.Br(),
                f"📊 Total Images: {dir_info['total_count']}", html.Br(),
                f"🎯 Mode: {mode.title()} Model Evaluation"
            ], className="mb-0")
        ], color="success")
        
        return analysis_content, False


@app.callback(
    [Output("mode-selector", "value", allow_duplicate=True),
     Output("model1-path-input", "value", allow_duplicate=True),
     Output("model2-path-input", "value", allow_duplicate=True),
     Output("test-dir-compare-input", "value", allow_duplicate=True)],
    Input("quick-compare-button", "n_clicks"),
    [State("model-path-input", "value"),
     State("test-dir-input", "value")],
    prevent_initial_call=True
)
def quick_compare_setup(n_clicks, current_model, current_test_dir):
    if n_clicks:
        return "compare", current_model or "models/best1.onnx", "models/best0.onnx", current_test_dir or "test/testing_on_real/val"
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update





@app.callback(
    [Output("single-model-inputs", "style"),
     Output("comparison-model-inputs", "style"),
     Output("quick-compare-button", "style")],
    Input("mode-selector", "value")
)
def toggle_input_mode(mode):
    if mode == "compare":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    else:
        return {"display": "block"}, {"display": "none"}, {"display": "inline-block"}


@app.callback(
    [Output("single-results", "style"),
     Output("comparison-results", "style")],
    Input("mode-selector", "value")
)
def toggle_results_mode(mode):
    if mode == "compare":
        return {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("model-path-modal", "is_open"),
    [Input("add-model2-path", "n_clicks"),
     Input("enable-comparison", "n_clicks"), 
     Input("modal-cancel", "n_clicks"),
     Input("modal-save", "n_clicks")],
    State("model-path-modal", "is_open")
)
def toggle_modal(add_clicks, enable_clicks, cancel_clicks, save_clicks, is_open):
    if not any([add_clicks, enable_clicks, cancel_clicks, save_clicks]):
        return False
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return False
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id in ["add-model2-path", "enable-comparison"]:
        return True
    elif button_id in ["modal-cancel", "modal-save"]:
        return False
    
    return is_open


@app.callback(
    [Output("modal-current-config", "children"),
     Output("model2-path-input", "value", allow_duplicate=True)],
    [Input("model-path-modal", "is_open"),
     Input("modal-save", "n_clicks")],
    [State("model-path-input", "value"),
     State("test-dir-input", "value"),
     State("modal-model2-path", "value"),
     State("mode-selector", "value")],
    prevent_initial_call=True
)
def update_modal_content(is_open, save_clicks, model1_path, test_dir, modal_model2, current_mode):
    if not is_open:
        return "", dash.no_update
    
    
    config_display = [
        html.P(f"Model 1: {model1_path or 'Not set'}", className="mb-1"),
        html.P(f"Test Directory: {test_dir or 'Not set'}", className="mb-1"),
        html.P(f"Current Mode: {current_mode}", className="mb-1")
    ]
    
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]["prop_id"].split(".")[0] == "modal-save":
        return config_display, modal_model2 or "models/best0.onnx"
    
    return config_display, dash.no_update


@app.callback(
    [Output("mode-selector", "value", allow_duplicate=True),
     Output("model1-path-input", "value", allow_duplicate=True),
     Output("test-dir-compare-input", "value", allow_duplicate=True)],
    Input("modal-save", "n_clicks"),
    [State("model-path-input", "value"),
     State("test-dir-input", "value"),
     State("modal-model2-path", "value")],
    prevent_initial_call=True
)
def switch_to_comparison_mode(save_clicks, model1_path, test_dir, model2_path):
    if save_clicks:
        return "compare", model1_path or "models/best1.onnx", test_dir or "test/testing_on_real/val"
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("current-models-display", "children"),
    [Input("model-path-input", "value"),
     Input("model1-path-input", "value"),
     Input("model2-path-input", "value"),
     Input("mode-selector", "value")]
)
def update_models_display(single_model, model1, model2, mode):
    if mode == "single":
        return f"Single: {single_model or 'Not set'}"
    else:
        return f"Model 1: {model1 or 'Not set'} | Model 2: {model2 or 'Not set'}"



@app.callback(
    [Output("results-section", "style"),
     Output("accuracy-card", "children"),
     Output("precision-card", "children"),
     Output("recall-card", "children"),
     Output("f1-card", "children"),
     Output("confusion-matrix-graph", "figure"),
     Output("roc-curve-graph", "figure"),
     Output("pr-curve-graph", "figure"),
     Output("predictions-table", "children"),
     Output("accuracy-comparison-card", "children"),
     Output("precision-comparison-card", "children"),
     Output("recall-comparison-card", "children"),
     Output("f1-comparison-card", "children"),
     Output("cm1-graph", "figure"),
     Output("cm2-graph", "figure"),
     Output("roc-comparison-graph", "figure"),
     Output("pr-comparison-graph", "figure"),
     Output("detailed-comparison-table", "children")],
    Input("evaluate-button", "n_clicks"),
    [State("mode-selector", "value"),
     State("model-path-input", "value"),
     State("test-dir-input", "value"),
     State("model1-path-input", "value"),
     State("model2-path-input", "value"),
     State("test-dir-compare-input", "value")],
    prevent_initial_call=True
)
def run_evaluation(n_clicks, mode, model_path, test_dir, model1_path, model2_path, test_dir_compare):
    empty_fig = go.Figure()
    empty_card = create_metric_card("", "N/A")
    
    if mode == "single":
        
        if not model_path or not test_dir:
            raise PreventUpdate
        
        print(f"Starting single model evaluation...")
        print(f"Model: {model_path}")
        print(f"Test Directory: {test_dir}")
        
        results, message = evaluate_model_on_custom_set(model_path.strip(), test_dir.strip())
        
        if results is None:
            empty_fig.update_layout(
                title="Evaluation Failed",
                annotations=[dict(text=f"Error: {message}", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            error_card = create_metric_card("Error", "N/A", color="danger")
            return {"display": "block"}, error_card, error_card, error_card, error_card, \
                   empty_fig, empty_fig, empty_fig, dbc.Alert(f"Error: {message}", color="danger"), \
                   empty_card, empty_card, empty_card, empty_card, empty_fig, empty_fig, empty_fig, empty_fig, html.Div()
        
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        model_name = os.path.basename(model_path)
        test_name = os.path.basename(test_dir)
        
        accuracy_card = create_metric_card("Accuracy", results["accuracy"], f"Model: {model_name}")
        precision_card = create_metric_card("Precision", results["precision"], f"Test: {test_name}")
        recall_card = create_metric_card("Recall", results["recall"], f"Time: {timestamp}")
        f1_card = create_metric_card("F1 Score", results["f1"], f"Images: {results['directory_info']['total_count']}")
        
        
        cm = np.array(results["confusion_matrix"])
        cm_fig = ff.create_annotated_heatmap(z=cm, x=["Predicted Undamaged", "Predicted Damaged"],
                                            y=["Actual Undamaged", "Actual Damaged"], annotation_text=cm, colorscale="Blues")
        cm_fig.update_layout(title=f"Confusion Matrix - {test_name}", height=400, margin=dict(l=50, r=50, t=80, b=50))
        
        fpr, tpr = results["roc_curve"]["fpr"], results["roc_curve"]["tpr"]
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {results["roc_auc"]:.3f})', line=dict(width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash', color='gray')))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=400, margin=dict(l=50, r=50, t=80, b=50))
        
        precision, recall = results["pr_curve"]["precision"], results["pr_curve"]["recall"]
        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall Curve', line=dict(width=3)))
        pr_fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=400, margin=dict(l=50, r=50, t=80, b=50))
        
        
        predictions_sample = results["prediction_details"][:20]
        table_rows = []
        for pred in predictions_sample:
            color = "success" if pred['correct'] else "danger"
            icon = "✅" if pred['correct'] else "❌"
            row = dbc.ListGroupItem([
                html.Div([
                    html.Strong(f"{icon} {pred['file']}"),
                    html.Br(),
                    html.Small(f"True: {pred['true_class']} | Pred: {pred['pred_class']} | Conf: {pred['confidence']:.3f}")
                ])
            ], color=color, style={"margin-bottom": "2px"})
            table_rows.append(row)
        
        predictions_table = html.Div([
            dbc.ListGroup(table_rows),
            html.Hr(),
            html.Small(f"Showing first 20 of {len(results['prediction_details'])} predictions", className="text-muted")
        ])
        
        return {"display": "block"}, accuracy_card, precision_card, recall_card, f1_card, \
               cm_fig, roc_fig, pr_fig, predictions_table, \
               empty_card, empty_card, empty_card, empty_card, empty_fig, empty_fig, empty_fig, empty_fig, html.Div()
               
    else:
        
        if not model1_path or not model2_path or not test_dir_compare:
            raise PreventUpdate
        
        print(f"Starting comparison evaluation...")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Test Directory: {test_dir_compare}")
        
        
        print("Evaluating Model 1...")
        results1, message1 = evaluate_model_on_custom_set(model1_path.strip(), test_dir_compare.strip())
        print("Evaluating Model 2...")
        results2, message2 = evaluate_model_on_custom_set(model2_path.strip(), test_dir_compare.strip())
        
        if results1 is None or results2 is None:
            error_msg = f"Model 1 Error: {message1}" if results1 is None else f"Model 2 Error: {message2}"
            empty_fig.update_layout(
                title="Comparison Failed",
                annotations=[dict(text=f"Error: {error_msg}", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
            )
            error_card = create_metric_card("Error", "N/A", color="danger")
            return {"display": "block"}, empty_card, empty_card, empty_card, empty_card, \
                   empty_fig, empty_fig, empty_fig, html.Div(), \
                   error_card, error_card, error_card, error_card, empty_fig, empty_fig, empty_fig, empty_fig, \
                   dbc.Alert(f"Error: {error_msg}", color="danger")
        
        
        model1_name = os.path.basename(model1_path)
        model2_name = os.path.basename(model2_path)
        test_name = os.path.basename(test_dir_compare)
        
        
        accuracy_comp = create_comparison_card("Accuracy", results1["accuracy"], results2["accuracy"], model1_name, model2_name)
        precision_comp = create_comparison_card("Precision", results1["precision"], results2["precision"], model1_name, model2_name)
        recall_comp = create_comparison_card("Recall", results1["recall"], results2["recall"], model1_name, model2_name)
        f1_comp = create_comparison_card("F1 Score", results1["f1"], results2["f1"], model1_name, model2_name)
        
        
        cm1 = np.array(results1["confusion_matrix"])
        cm2 = np.array(results2["confusion_matrix"])
        
        cm1_fig = ff.create_annotated_heatmap(z=cm1, x=["Pred Undmg", "Pred Dmg"], y=["Act Undmg", "Act Dmg"], 
                                             annotation_text=cm1, colorscale="Blues")
        cm1_fig.update_layout(title=f"{model1_name}", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        
        cm2_fig = ff.create_annotated_heatmap(z=cm2, x=["Pred Undmg", "Pred Dmg"], y=["Act Undmg", "Act Dmg"], 
                                             annotation_text=cm2, colorscale="Reds")
        cm2_fig.update_layout(title=f"{model2_name}", height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        
        
        roc_comp_fig = go.Figure()
        roc_comp_fig.add_trace(go.Scatter(x=results1["roc_curve"]["fpr"], y=results1["roc_curve"]["tpr"], 
                                         mode='lines', name=f'{model1_name} (AUC = {results1["roc_auc"]:.3f})', 
                                         line=dict(color='blue', width=3)))
        roc_comp_fig.add_trace(go.Scatter(x=results2["roc_curve"]["fpr"], y=results2["roc_curve"]["tpr"], 
                                         mode='lines', name=f'{model2_name} (AUC = {results2["roc_auc"]:.3f})', 
                                         line=dict(color='red', width=3)))
        roc_comp_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', 
                                         line=dict(dash='dash', color='gray')))
        roc_comp_fig.update_layout(title="ROC Curves Comparison", xaxis_title="False Positive Rate", 
                                  yaxis_title="True Positive Rate", height=400, margin=dict(l=50, r=50, t=80, b=50))
        
        
        pr_comp_fig = go.Figure()
        pr_comp_fig.add_trace(go.Scatter(x=results1["pr_curve"]["recall"], y=results1["pr_curve"]["precision"], 
                                        mode='lines', name=f'{model1_name}', line=dict(color='blue', width=3)))
        pr_comp_fig.add_trace(go.Scatter(x=results2["pr_curve"]["recall"], y=results2["pr_curve"]["precision"], 
                                        mode='lines', name=f'{model2_name}', line=dict(color='red', width=3)))
        pr_comp_fig.update_layout(title="Precision-Recall Curves Comparison", xaxis_title="Recall", 
                                 yaxis_title="Precision", height=400, margin=dict(l=50, r=50, t=80, b=50))
        
        
        comparison_data = [
            ["Metric", model1_name, model2_name, "Difference", "Winner"],
            ["Accuracy", f"{results1['accuracy']*100:.2f}%", f"{results2['accuracy']*100:.2f}%", 
             f"{(results1['accuracy']-results2['accuracy'])*100:+.2f}%", 
             model1_name if results1['accuracy'] > results2['accuracy'] else model2_name],
            ["Precision", f"{results1['precision']*100:.2f}%", f"{results2['precision']*100:.2f}%", 
             f"{(results1['precision']-results2['precision'])*100:+.2f}%", 
             model1_name if results1['precision'] > results2['precision'] else model2_name],
            ["Recall", f"{results1['recall']*100:.2f}%", f"{results2['recall']*100:.2f}%", 
             f"{(results1['recall']-results2['recall'])*100:+.2f}%", 
             model1_name if results1['recall'] > results2['recall'] else model2_name],
            ["F1 Score", f"{results1['f1']*100:.2f}%", f"{results2['f1']*100:.2f}%", 
             f"{(results1['f1']-results2['f1'])*100:+.2f}%", 
             model1_name if results1['f1'] > results2['f1'] else model2_name],
            ["ROC AUC", f"{results1['roc_auc']:.4f}", f"{results2['roc_auc']:.4f}", 
             f"{results1['roc_auc']-results2['roc_auc']:+.4f}", 
             model1_name if results1['roc_auc'] > results2['roc_auc'] else model2_name],
        ]
        
        comparison_table = html.Div([
            dbc.Table.from_dataframe(
                pd.DataFrame(comparison_data[1:], columns=comparison_data[0]), 
                striped=True, bordered=True, hover=True, responsive=True, className="mt-3"
            ),
            html.Hr(),
            html.Div([
                html.H6("Summary:", className="fw-bold"),
                html.P(f"Test set: {results1['directory_info']['total_count']} images ({results1['directory_info']['damaged_count']} damaged, {results1['directory_info']['undamaged_count']} undamaged)", className="mb-1"),
                html.P(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", className="mb-0 text-muted")
            ])
        ])
        
        return {"display": "block"}, empty_card, empty_card, empty_card, empty_card, \
               empty_fig, empty_fig, empty_fig, html.Div(), \
               accuracy_comp, precision_comp, recall_comp, f1_comp, \
               cm1_fig, cm2_fig, roc_comp_fig, pr_comp_fig, comparison_table

@app.callback(
    Output("export-results", "n_clicks"),
    Input("export-results", "n_clicks"),
    prevent_initial_call=True
)
def export_results(n_clicks):
    if n_clicks:
        
        print("Export functionality - to be implemented")
    return 0

@app.callback(
    Output("save-config", "n_clicks"),
    Input("save-config", "n_clicks"),
    prevent_initial_call=True
)
def save_config(n_clicks):
    if n_clicks:
        
        print("Save config functionality - to be implemented")
    return 0

@app.callback(
    Output("load-config", "n_clicks"),
    Input("load-config", "n_clicks"),
    prevent_initial_call=True
)
def load_config(n_clicks):
    if n_clicks:
        
        print("Load config functionality - to be implemented")
    return 0


@app.callback(
    Output("browse-model1", "n_clicks"),
    Input("browse-model1", "n_clicks"),
    prevent_initial_call=True
)
def browse_model1(n_clicks):
    if n_clicks:
        print("Browse model 1 - file dialog functionality to be implemented")
    return 0

@app.callback(
    Output("browse-test-dir", "n_clicks"),
    Input("browse-test-dir", "n_clicks"),
    prevent_initial_call=True
)
def browse_test_dir(n_clicks):
    if n_clicks:
        print("Browse test directory - file dialog functionality to be implemented")
    return 0

@app.callback(
    Output("browse-model1-comp", "n_clicks"),
    Input("browse-model1-comp", "n_clicks"),
    prevent_initial_call=True
)
def browse_model1_comp(n_clicks):
    if n_clicks:
        print("Browse model 1 comparison - file dialog functionality to be implemented")
    return 0

@app.callback(
    Output("browse-test-dir-comp", "n_clicks"),
    Input("browse-test-dir-comp", "n_clicks"),
    prevent_initial_call=True
)
def browse_test_dir_comp(n_clicks):
    if n_clicks:
        print("Browse test directory comparison - file dialog functionality to be implemented")
    return 0


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Enhanced Custom Test Set Evaluation Dashboard")
    print("=" * 60)
    print("📊 Features:")
    print("   • Single model evaluation")
    print("   • Two model comparison")
    print("   • Interactive path selection")
    print("   • Comprehensive metrics and visualizations")
    print("   • Real-time directory analysis")
    print("=" * 60)
    print("🌐 Navigate to: http://127.0.0.1:8050")
    print("📁 Example usage:")
    print("   • Model path: models/best1.onnx")
    print("   • Test directory: test/testing_on_real/val")
    print("=" * 60)
    
    app.run(debug=True, port=8050)