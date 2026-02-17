from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io, base64, os, datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global variable for ML model
ml_model = None
feature_importance = None
model_accuracy = 0.0

# ── FEATURES ─────────────────────────────
FEATURES = ['age_years','sex','height','weight','bmi',
            'ap_hi','ap_lo','cholesterol','gluc',
            'smoke','alco','active','overweight']

# DATA HELPERS
# ============================================================================

def load_medical_data(filepath='data/medical_examination.csv'):
    try:
        df = pd.read_csv(filepath)
        # Accept both 'sex' and 'gender' column names
        if 'gender' in df.columns and 'sex' not in df.columns:
            df.rename(columns={'gender': 'sex'}, inplace=True)
        return prepare_medical_data(df)
    except FileNotFoundError:
        return pd.DataFrame()


def prepare_medical_data(df):
    df = df.copy()
    h_m = df['height'] / 100
    df['bmi']        = df['weight'] / (h_m ** 2)
    df['overweight'] = (df['bmi'] > 25).astype(int)
    df['age_years']  = df['age'] / 365
    df['cholesterol_norm'] = np.where(df['cholesterol'] == 1, 0, 1)
    df['gluc_norm']        = np.where(df['gluc']        == 1, 0, 1)

    df['bp_category'] = pd.cut(
        df['ap_hi'], bins=[0,120,140,180,300],
        labels=['Normal','Elevated','High','Crisis']
    )
    risk = (
        df['overweight']      * 15 +
        df['cholesterol_norm']* 20 +
        df['gluc_norm']       * 20 +
        df['smoke']           * 15 +
        (1 - df['active'])    * 10 +
        df['alco']            * 10 +
        ((df['age_years']-30)/40*10).clip(0,10)
    )
    df['risk_score']    = risk.clip(0, 100)
    df['risk_category'] = pd.cut(
        df['risk_score'], bins=[0,30,50,70,100],
        labels=['Low','Moderate','High','Critical']
    )
    return df


def clean_data(df):
    return df[
        (df['ap_lo']  <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


# ML MODEL
# ============================================================================

def train_model(df=None):
    global ml_model, feature_importance, model_accuracy
    if df is None:
        df = load_medical_data()
    if df.empty:
        return

    df_c  = clean_data(df)
    valid = [f for f in FEATURES if f in df_c.columns]
    X, y  = df_c[valid], df_c['cardio']

    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42)
    ml_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ml_model.fit(Xtr, ytr)

    model_accuracy    = accuracy_score(yte, ml_model.predict(Xte))
    feature_importance = dict(zip(valid, ml_model.feature_importances_))
    print(f"✓ ML Model trained  accuracy={model_accuracy:.2%}")


# VISUALISATIONS
# ============================================================================

def viz_categorical(df):
    df_m = pd.melt(df, id_vars=['cardio'],
                   value_vars=['cholesterol_norm','gluc_norm',
                               'smoke','alco','active','overweight'])
    df_m = df_m.groupby(['cardio','variable','value']).size().reset_index(name='total')

    g = sns.catplot(x='variable', y='total', hue='value', col='cardio',
                    data=df_m, kind='bar', height=5, aspect=1.3,
                    palette=['#26c6da','#1e88e5'])
    g.set_axis_labels("Health Factor", "Patient Count")
    g.set_titles("Cardiovascular Disease = {col_name}")

    buf = io.BytesIO()
    g.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close('all')
    return b64


def viz_heatmap(df):
    df_c   = clean_data(df)
    # ONLY numeric columns — avoids the corr() FutureWarning / error
    num_cols = [c for c in FEATURES + ['cardio','risk_score'] if c in df_c.columns]
    corr   = df_c[num_cols].corr()
    mask   = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                square=True, linewidths=0.5,
                cmap='coolwarm', center=0,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('Medical Data — Correlation Matrix', fontsize=15, pad=18)
    fig.tight_layout()
    return fig_to_b64(fig)


def viz_risk_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for cv, col, lbl in [(0,'#26c6da','No Disease'),(1,'#ef5350','Has Disease')]:
        axes[0].hist(df[df['cardio']==cv]['risk_score'],
                     bins=30, alpha=0.65, color=col, label=lbl)
    axes[0].set_title('Risk Score Distribution')
    axes[0].set_xlabel('Risk Score (0–100)')
    axes[0].set_ylabel('Patients')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    bp = axes[1].boxplot(
        [df[df['cardio']==0]['risk_score'], df[df['cardio']==1]['risk_score']],
        labels=['No Disease','Has Disease'], patch_artist=True
    )
    bp['boxes'][0].set_facecolor('#26c6da')
    bp['boxes'][1].set_facecolor('#ef5350')
    axes[1].set_title('Risk Score by Cardio Status')
    axes[1].set_ylabel('Risk Score')
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig_to_b64(fig)


# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/statistics')
def get_statistics():
    try:
        df = load_medical_data()
        n  = len(df)
        return jsonify({
            'total_patients':              n,
            'cardio_percentage':           round(df['cardio'].mean()*100, 1),
            'avg_age':                     round(df['age_years'].mean(), 1),
            'avg_bmi':                     round(df['bmi'].mean(), 1),
            'overweight_percentage':       round(df['overweight'].mean()*100, 1),
            'smoking_percentage':          round(df['smoke'].mean()*100, 1),
            'active_percentage':           round(df['active'].mean()*100, 1),
            'high_cholesterol_percentage': round(df['cholesterol_norm'].mean()*100, 1),
            'high_glucose_percentage':     round(df['gluc_norm'].mean()*100, 1),
            'sex_distribution':            df['sex'].value_counts().to_dict(),
            'risk_distribution':           df['risk_category'].value_counts().to_dict(),
            'bp_distribution':             df['bp_category'].value_counts().to_dict(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/risk-analysis')
def get_risk_analysis():
    try:
        df  = load_medical_data()
        out = []
        for cat in ['Low','Moderate','High','Critical']:
            s = df[df['risk_category'] == cat]
            if len(s):
                out.append({
                    'category':    cat,
                    'count':       int(len(s)),
                    'percentage':  round(len(s)/len(df)*100, 1),
                    'cardio_rate': round(s['cardio'].mean()*100, 1),
                    'avg_age':     round(s['age_years'].mean(), 1),
                    'avg_bmi':     round(s['bmi'].mean(), 1),
                })
        return jsonify(out)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/lifestyle-factors')
def lifestyle_factors():
    try:
        df = load_medical_data()
        return jsonify({
            'smoking':     {'total': int(df['smoke'].sum()),
                            'cardio_rate': round(df[df['smoke']==1]['cardio'].mean()*100,1)},
            'alcohol':     {'total': int(df['alco'].sum()),
                            'cardio_rate': round(df[df['alco']==1]['cardio'].mean()*100,1)},
            'active':      {'total': int(df['active'].sum()),
                            'cardio_rate': round(df[df['active']==1]['cardio'].mean()*100,1)},
            'overweight':  {'total': int(df['overweight'].sum()),
                            'cardio_rate': round(df[df['overweight']==1]['cardio'].mean()*100,1)},
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Visualisation endpoints ────────────────────────────────────────────────

@app.route('/api/visualizations/categorical')
def api_cat():
    try:
        df = load_medical_data()
        return jsonify({'image': f'data:image/png;base64,{viz_categorical(df)}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualizations/heatmap')
def api_heatmap():
    try:
        df = load_medical_data()
        return jsonify({'image': f'data:image/png;base64,{viz_heatmap(df)}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/visualizations/risk-distribution')
def api_risk_dist():
    try:
        df = load_medical_data()
        return jsonify({'image': f'data:image/png;base64,{viz_risk_distribution(df)}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── ML model info ──────────────────────────────────────────────────────────

@app.route('/api/ml-model-info')
def api_ml_info():
    try:
        if ml_model is None:
            train_model()
        return jsonify({
            'trained':            ml_model is not None,
            'accuracy':           round(model_accuracy * 100, 1),
            'feature_importance': feature_importance,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# RISK PREDICTOR
# ============================================================================

@app.route('/api/predict-risk', methods=['POST'])
def api_predict():
    try:
        raw = request.get_json()

        # Age: frontend sends years directly
        age_years = float(raw.get('age', 40))

        # Sex: accept both 'sex' and 'gender'
        sex = int(raw.get('sex', raw.get('gender', 1)))

        height = float(raw.get('height', 170))
        weight = float(raw.get('weight', 70))
        bmi    = weight / ((height/100) ** 2)

        patient = {
            'age_years':   age_years,
            'sex':         sex,
            'height':      height,
            'weight':      weight,
            'bmi':         bmi,
            'ap_hi':       float(raw.get('ap_hi', 120)),
            'ap_lo':       float(raw.get('ap_lo', 80)),
            'cholesterol': int(raw.get('cholesterol', 1)),
            'gluc':        int(raw.get('gluc', 1)),
            'smoke':       int(raw.get('smoke', 0)),
            'alco':        int(raw.get('alco', 0)),
            'active':      int(raw.get('active', 1)),
            'overweight':  1 if bmi > 25 else 0,
        }

        if ml_model is None:
            train_model()

        valid = [f for f in FEATURES if f in patient]
        X     = pd.DataFrame([patient])[valid]
        pred  = ml_model.predict(X)[0]
        proba = ml_model.predict_proba(X)[0]

        return jsonify({
            'has_risk':         bool(pred),
            'risk_probability': round(float(proba[1]) * 100, 1),
            'confidence':       round(float(max(proba)) * 100, 1),
        })

    except Exception as e:
        print(f"[predict-risk] ERROR: {e}")
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# UPLOAD DATA
# ============================================================================

@app.route('/api/upload', methods=['POST'])
def api_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not f.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400

        df = pd.read_csv(f)

        # Rename gender → sex
        if 'gender' in df.columns and 'sex' not in df.columns:
            df.rename(columns={'gender': 'sex'}, inplace=True)

        required = ['age','height','weight','ap_hi','ap_lo',
                    'cholesterol','gluc','smoke','alco','active','cardio','sex']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {", ".join(missing)}'}), 400

        df = prepare_medical_data(df)

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        df.to_csv(save_path, index=False)

        # Retrain with new data
        train_model(df)

        return jsonify({
            'success': True,
            'message': f'Loaded {len(df):,} patient records and retrained model',
        })

    except Exception as e:
        print(f"[upload] ERROR: {e}")
        return jsonify({'error': str(e)}), 500


# PDF EXPORT
# ============================================================================

@app.route('/api/export/pdf', methods=['GET','POST'])
def api_export_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Image as RLImage, Table, TableStyle,
                                        HRFlowable)
        from reportlab.lib.units import inch, cm

        df  = load_medical_data()
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=1.5*cm, bottomMargin=1.5*cm,
                                leftMargin=2*cm,  rightMargin=2*cm)

        styles = getSampleStyleSheet()
        BLUE   = colors.HexColor('#1e88e5')
        NAVY   = colors.HexColor('#0d47a1')
        GRAY   = colors.HexColor('#455a64')
        LGRAY  = colors.HexColor('#90a4ae')

        T = ParagraphStyle('T', parent=styles['Title'],  fontSize=22,
                           textColor=NAVY, fontName='Helvetica-Bold', spaceAfter=4)
        S = ParagraphStyle('S', parent=styles['Normal'], fontSize=11,
                           textColor=BLUE, spaceAfter=18)
        H = ParagraphStyle('H', parent=styles['Heading2'], fontSize=13,
                           textColor=NAVY, fontName='Helvetica-Bold',
                           spaceBefore=14, spaceAfter=6)
        B = ParagraphStyle('B', parent=styles['Normal'], fontSize=10,
                           textColor=GRAY, spaceAfter=5)
        F = ParagraphStyle('F', parent=styles['Normal'], fontSize=8,
                           textColor=LGRAY, alignment=1)

        story = []

        # Header
        story += [
            Paragraph("MediVision Pro", T),
            Paragraph("Medical Data Analytics — Full Report", S),
            Paragraph(f"Generated: {datetime.datetime.now().strftime('%B %d, %Y  %H:%M')}", B),
            HRFlowable(width="100%", thickness=2, color=BLUE),
            Spacer(1, 0.25*inch),
        ]

        # ── Key Stats table ────────────────────────────────────────────────
        story.append(Paragraph("Key Statistics", H))
        n   = len(df)
        kpi = [
            ['Total Patients',       f'{n:,}',
             'Average Age',          f'{round(df["age_years"].mean(),1)} yrs'],
            ['Cardiovascular Rate',  f'{round(df["cardio"].mean()*100,1)}%',
             'Average BMI',          str(round(df["bmi"].mean(),1))],
            ['Overweight Rate',      f'{round(df["overweight"].mean()*100,1)}%',
             'Physically Active',    f'{round(df["active"].mean()*100,1)}%'],
            ['Smoking Rate',         f'{round(df["smoke"].mean()*100,1)}%',
             'High Cholesterol',     f'{round(df["cholesterol_norm"].mean()*100,1)}%'],
        ]
        hdr = [['Metric','Value','Metric','Value']]
        tbl = Table(hdr + kpi, colWidths=[4.5*cm, 3.5*cm, 4.5*cm, 3.5*cm])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), BLUE),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
            ('ALIGN',      (0,0),(-1,-1),'CENTER'),
            ('FONTSIZE',   (0,0),(-1,-1), 10),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#e3f2fd'), colors.white]),
            ('GRID',       (0,0),(-1,-1), 0.4, colors.HexColor('#b0bec5')),
            ('ROWHEIGHT',  (0,0),(-1,-1), 22),
        ]))
        story += [tbl, Spacer(1, 0.25*inch)]

        # ── Risk table ─────────────────────────────────────────────────────
        story.append(Paragraph("Risk Category Distribution", H))
        rh   = [['Category','Patients','% of Total','Cardio Rate']]
        rdata = rh
        for cat in ['Low','Moderate','High','Critical']:
            s = df[df['risk_category']==cat]
            if len(s):
                rdata.append([
                    cat, f'{len(s):,}',
                    f'{round(len(s)/n*100,1)}%',
                    f'{round(s["cardio"].mean()*100,1)}%',
                ])
        rt = Table(rdata, colWidths=[4*cm,4*cm,4*cm,4*cm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0), NAVY),
            ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
            ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
            ('ALIGN',      (0,0),(-1,-1),'CENTER'),
            ('FONTSIZE',   (0,0),(-1,-1), 10),
            ('GRID',       (0,0),(-1,-1), 0.4, colors.HexColor('#b0bec5')),
            ('ROWHEIGHT',  (0,0),(-1,-1), 22),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#e8eaf6'), colors.white]),
        ]))
        story += [rt, Spacer(1, 0.25*inch)]

        # ── Helper embed figure ────────────────────────────────────────────
        def embed(fig, w=6.2*inch):
            b = io.BytesIO()
            fig.savefig(b, format='png', dpi=130, bbox_inches='tight')
            b.seek(0); plt.close(fig)
            img = RLImage(b)
            ratio = img.imageWidth / img.imageHeight
            img.drawWidth  = w
            img.drawHeight = w / ratio
            story.append(img)
            story.append(Spacer(1, 0.2*inch))

        # ── Categorical chart ──────────────────────────────────────────────
        story.append(Paragraph("Health Factors Comparison", H))
        story.append(Paragraph(
            "Count of good (0) vs bad (1) outcomes for each health factor, "
            "grouped by cardiovascular disease status.", B))
        df_m = pd.melt(df, id_vars=['cardio'],
                       value_vars=['cholesterol_norm','gluc_norm',
                                   'smoke','alco','active','overweight'])
        df_m = df_m.groupby(['cardio','variable','value']).size().reset_index(name='total')
        g = sns.catplot(x='variable',y='total',hue='value',col='cardio',
                        data=df_m,kind='bar',height=4,aspect=1.15,
                        palette=['#26c6da','#1e88e5'])
        g.set_axis_labels("Factor","Count")
        g.set_titles("Cardio = {col_name}")
        cb = io.BytesIO()
        g.savefig(cb, format='png', dpi=130, bbox_inches='tight')
        cb.seek(0); plt.close('all')
        ci = RLImage(cb)
        ci.drawWidth=6.2*inch; ci.drawHeight=3.2*inch
        story += [ci, Spacer(1,0.2*inch)]

        # ── Risk distribution ──────────────────────────────────────────────
        story.append(Paragraph("Risk Score Distribution", H))
        fig2, axes = plt.subplots(1, 2, figsize=(11,4))
        for cv,cl,lb in [(0,'#26c6da','No Disease'),(1,'#ef5350','Has Disease')]:
            axes[0].hist(df[df['cardio']==cv]['risk_score'],
                         bins=30,alpha=0.65,color=cl,label=lb)
        axes[0].set_title('Histogram'); axes[0].set_xlabel('Risk Score')
        axes[0].legend(); axes[0].grid(alpha=0.3)
        bp2 = axes[1].boxplot(
            [df[df['cardio']==0]['risk_score'], df[df['cardio']==1]['risk_score']],
            labels=['No Disease','Has Disease'], patch_artist=True)
        bp2['boxes'][0].set_facecolor('#26c6da')
        bp2['boxes'][1].set_facecolor('#ef5350')
        axes[1].set_title('Box Plot'); axes[1].grid(alpha=0.3)
        fig2.tight_layout()
        embed(fig2)

        # ── Heatmap ────────────────────────────────────────────────────────
        story.append(Paragraph("Correlation Heatmap", H))
        story.append(Paragraph(
            "Pearson correlations between all numeric medical variables. "
            "Red = positive, Blue = negative.", B))
        df_c   = clean_data(df)
        nc     = [c for c in FEATURES+['cardio','risk_score'] if c in df_c.columns]
        corr   = df_c[nc].corr()
        mask   = np.triu(np.ones_like(corr, dtype=bool))
        fig3, ax3 = plt.subplots(figsize=(13, 10))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', square=True,
                    linewidths=0.5, cmap='coolwarm', center=0,
                    cbar_kws={'shrink':0.8}, ax=ax3)
        ax3.set_title('Correlation Matrix', fontsize=14, pad=14)
        fig3.tight_layout()
        embed(fig3, w=5.8*inch)

        # ── Clinical recommendations ───────────────────────────────────────
        story.append(Paragraph("Clinical Recommendations", H))
        recs = [
            f"<b>Blood Pressure:</b> {len(df[df['ap_hi']>=140]):,} patients have "
            f"systolic BP ≥ 140 mmHg — require monitoring.",
            f"<b>Weight Management:</b> {int(df['overweight'].sum()):,} patients "
            f"({round(df['overweight'].mean()*100,1)}%) are overweight (BMI > 25).",
            f"<b>Cholesterol:</b> {int(df['cholesterol_norm'].sum()):,} patients "
            f"have above-normal cholesterol.",
            f"<b>High-Risk Patients:</b> "
            f"{len(df[df['risk_category'].isin(['High','Critical'])]):,} patients "
            f"need urgent follow-up.",
            f"<b>Lifestyle:</b> Promote physical activity "
            f"(only {round(df['active'].mean()*100,1)}% currently active) "
            f"and smoking cessation.",
        ]
        for r in recs:
            story.append(Paragraph(f"• {r}", B))

        # Footer
        story += [
            Spacer(1, 0.3*inch),
            HRFlowable(width="100%", thickness=1, color=BLUE),
            Spacer(1, 0.1*inch),
            Paragraph("MediVision Pro | Medical Data Analytics | "
                       "freeCodeCamp Data Analysis with Python", F),
        ]

        doc.build(story)
        buf.seek(0)
        fname = f'MediVision_Report_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        return send_file(buf, mimetype='application/pdf',
                         as_attachment=True, download_name=fname)

    except ImportError:
        return jsonify({
            'error': 'Missing library. Run:  pip install reportlab'
        }), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def e404(e): return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def e500(e): return jsonify({'error': 'Server error'}), 500


# RUN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥  MEDIVISION PRO")
    print("=" * 60)
    df0 = load_medical_data()
    if not df0.empty:
        print(f"✓  Loaded {len(df0):,} patient records")
        train_model(df0)
    else:
        print("⚠️   Add medical_examination.csv to data/")
    print("\n🚀  http://localhost:5000\n" + "="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)

