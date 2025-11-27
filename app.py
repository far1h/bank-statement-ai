import streamlit as st
import pandas as pd
import numpy as np
import spacy
import os
import tempfile
import re
import pdfplumber 
from tabula.io import read_pdf
from datetime import datetime
from io import BytesIO 

# --- CONFIGURATION ---
st.set_page_config(page_title="BCA Statement Processor", layout="wide")

# ==========================================
#          PASSWORD PROTECTION
# ==========================================
# CHANGE YOUR PASSWORD HERE
ACCEPTED_PASSWORD = "admin123"

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password_input"] == ACCEPTED_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]  # Clean up
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "Enter Password to Access:", 
            type="password", 
            on_change=password_entered, 
            key="password_input"
        )
        return False
    
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Enter Password to Access:", 
            type="password", 
            on_change=password_entered, 
            key="password_input"
        )
        st.error("😕 Password incorrect. Please try again.")
        return False
    
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # STOPS execution here if auth fails

# ==========================================
#          MAIN APP LOGIC
# ==========================================

# --- COORDINATE CONSTANTS (Easily Editable) ---
# Format: (top, left, bottom, right) for Tabula
HEADER_AREA = (70, 315, 141, 548)
TABLE_AREA = (231, 25, 797, 577)
COLUMNS_X = [86, 184, 300, 340, 467]

# --- LOAD NER MODEL (Cached) ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# --- HELPER FUNCTIONS ---

def clean_numeric_columns(dataframe, columns):
    for column in columns:
        dataframe[column] = dataframe[column].str.replace(',', '')
        dataframe[column] = pd.to_numeric(dataframe[column], errors='coerce')
        dataframe[column] = dataframe[column].astype('float')
    return dataframe

def union_source(dataframes):
    dfs = []
    for temp_df in dataframes:
        temp_df[['amount', 'type']] = temp_df[4].str.extract(r'([\d,]+(?:\.\d+)?)\s*(DB|CR)?')
        temp_df = temp_df.drop(temp_df.columns[4], axis=1)

        if(len(temp_df.columns) == 7):
            temp_df.columns = ['date', 'desc', 'detail', 'branch', 'balance', 'amount', 'type']
            temp_df = temp_df[['date', 'desc', 'detail', 'branch', 'amount', 'type', 'balance']]
            dfs.append(temp_df)

    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.fillna(value=np.nan)
    return df

def insert_shifted_column(dataframe):
    if dataframe.empty: return dataframe
    dataframe['prev_date'] = dataframe['date'].shift(1)
    dataframe['prev_desc'] = dataframe['desc'].shift(1)
    dataframe['prev_detail'] = dataframe['detail'].shift(1)
    dataframe['prev_branch'] = dataframe['branch'].shift(1)
    dataframe['prev_amount'] = dataframe['amount'].shift(1)
    dataframe['prev_transaction_type'] = dataframe['type'].shift(1)
    dataframe['prev_balance'] = dataframe['balance'].shift(1)
    dataframe = dataframe.fillna(value=np.nan)
    return dataframe

def extract_transactions(dataframe):
    if dataframe.empty: return dataframe
    transactions = []
    details = []
    descs = []
    temp = {}

    skip_keywords = ['SALDOAWAL', 'SALDOAKHIR', 'MUTASICR', 'MUTASIDB', 'TOTALMUTASI']

    for index, row in dataframe.iterrows():
        d_text = str(row['desc']) if not pd.isna(row['desc']) else ""
        det_text = str(row['detail']) if not pd.isna(row['detail']) else ""
        full_line_clean = (d_text + det_text).upper().replace(" ", "").replace(":", "")

        if any(keyword in full_line_clean for keyword in skip_keywords):
            continue

        if not pd.isna(row['amount']):
            if temp:
                combined_desc = ' '.join(descs) if descs else ''
                combined_detail = ' '.join(details) if details else ''
                final_desc = f"{combined_desc} {combined_detail}".strip()

                all_text_parts = descs + details
                if len(all_text_parts) >= 2:
                    tail_text = " ".join(all_text_parts[-2:])
                elif len(all_text_parts) == 1:
                    tail_text = all_text_parts[0]
                else:
                    tail_text = ""

                transaction = {
                    "date": temp['date'],
                    "desc": final_desc,
                    "desc_tail": tail_text,
                    "amount": temp['amount'],
                    "transaction_type": temp['transaction_type'] if temp['transaction_type'] == 'DB' else 'CR',
                    "balance": temp['balance']
                }
                transactions.append(transaction)
                details = []
                descs = []
                temp = {}

            temp = {
                'date': row['date'],
                'amount': row['amount'],
                'transaction_type': row['type'],
                'balance': row['balance']
            }

        if (not pd.isna(row['desc'])):
            if row['desc'] not in ['KETE', 'RANGAN']:
                descs.append(row['desc'])
        
        if (not pd.isna(row['detail'])):
            if row['detail'] not in ['KETE', 'RANGAN']:
                details.append(row['detail'])

    if temp:
        combined_desc = ' '.join(descs) if descs else ''
        combined_detail = ' '.join(details) if details else ''
        final_desc = f"{combined_desc} {combined_detail}".strip()
        
        all_text_parts = descs + details
        if len(all_text_parts) >= 2:
            tail_text = " ".join(all_text_parts[-2:])
        elif len(all_text_parts) == 1:
            tail_text = all_text_parts[0]
        else:
            tail_text = ""
        
        transaction = {
            "date": temp['date'],
            "desc": final_desc,
            "desc_tail": tail_text,
            "amount": temp['amount'],
            "transaction_type": temp['transaction_type'] if temp['transaction_type'] == 'DB' else 'CR',
            "balance": temp['balance']
        }
        transactions.append(transaction)

    return pd.DataFrame(transactions)

def apply_categories(dataframe):
    if dataframe.empty: return dataframe
    categories = []
    beneficiaries = [] 
    
    for index, row in dataframe.iterrows():
        desc_upper = str(row['desc']).upper()
        raw_tail_text = str(row.get('desc_tail', ''))
        raw_tail_text = " ".join(raw_tail_text.split())
        trans_type = row['transaction_type']
        cat = ""
        
        if desc_upper == "BIAYA ADM": cat = "administration charges".upper()
        elif desc_upper == "TARIKAN PEMINDAHAN": cat = "transfer withdrawal".upper()
        elif desc_upper.startswith("TARIKAN ATM"): cat = "atm withdrawal".upper()
        elif desc_upper.startswith("FLAZZ BCA TOPUP"): cat = "flazz top up".upper()
        elif desc_upper == "BUNGA": cat = "bunga".upper()   
        elif desc_upper == "SETORAN TUNAI": cat = "cash deposit".upper()
        elif desc_upper == "TARIKAN TUNAI": cat = "cash withdrawal".upper()
        elif desc_upper == "PAJAK BUNGA": cat = "pajak bunga".upper()
        elif desc_upper == "DR KOREKSI BUNGA": cat = "correction interest debit".upper()
        else:
            if trans_type == 'DB': cat = f"TRANSFERRED TO {raw_tail_text}"
            elif trans_type == 'CR': cat = f"TRANSFERRED FROM {raw_tail_text}"
        
        categories.append(cat)

        if cat.startswith("TRANSFERRED"):
            caps_matches = re.findall(r"\b[A-Z.'-]{2,}\b", raw_tail_text)
            cleaned_caps_text = " ".join(caps_matches)
            candidate_text = cleaned_caps_text if cleaned_caps_text else raw_tail_text
            found_entity = None
            
            if nlp:
                text_for_ner = candidate_text.title() 
                doc = nlp(text_for_ner)
                person_parts = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                if person_parts: found_entity = " ".join(person_parts).upper()
                if not found_entity:
                    org_parts = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                    if org_parts: found_entity = " ".join(org_parts).upper()
            
            if found_entity: beneficiaries.append(found_entity)
            else:
                if cleaned_caps_text: beneficiaries.append(cleaned_caps_text)
                else: beneficiaries.append(candidate_text.replace("-", "").lstrip())
        else:
            beneficiaries.append("-")
    
    dataframe['category'] = categories
    dataframe['beneficiary_name'] = beneficiaries
    dataframe['bank_name'] = 'BCA' 
    if 'desc_tail' in dataframe.columns: dataframe = dataframe.drop('desc_tail', axis=1)
    return dataframe

def calculate_balance(dataframe, init_balance):
    if dataframe.empty: return dataframe
    dataframe['balance'] = init_balance
    for index, row in dataframe.iterrows():
        if row['transaction_type'] == 'DB':
            if index == 0: dataframe.at[index, 'balance'] -= row['amount']
            else: dataframe.at[index, 'balance'] = dataframe.at[index - 1, 'balance'] - row['amount']
        elif row['transaction_type'] == 'CR':
            if index == 0: dataframe.at[index, 'balance'] += row['amount']
            else: dataframe.at[index, 'balance'] = dataframe.at[index - 1, 'balance'] + row['amount']
    return dataframe

def get_year_month(sheet_name):
    try:
        year, month_name = sheet_name.split(' ')
        month_dict = {'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4, 'MEI': 5, 'JUNI': 6, 'JULI': 7, 'AGUSTUS': 8, 'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12}
        return int(year), month_dict[month_name]
    except: return 0, 0 

# --- MAIN STREAMLIT APP ---

if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'excel_bytes' not in st.session_state: st.session_state.excel_bytes = None

st.title("📄 BCA Statement Processor")
st.markdown("Upload multiple PDF bank statements to parse, categorize, and merge them into a single table.")

uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# ==========================================
#      FIXED FEATURE: INSPECT COORDINATES
# ==========================================
with st.expander("🔎 Inspect PDF Coordinates (Debug)", expanded=True):
    st.info("Use this tool to verify the Red (Header), Blue (Table), and Green (Columns) boxes.")
    
    if uploaded_files:
        # 1. Select File
        selected_inspect_file = st.selectbox(
            "Select file to inspect:", 
            uploaded_files, 
            format_func=lambda x: x.name,
            key="inspect_file_select"
        )
        
        if selected_inspect_file:
            try:
                with pdfplumber.open(selected_inspect_file) as pdf:
                    # 2. Select Page
                    total_pages = len(pdf.pages)
                    page_options = list(range(1, total_pages + 1))
                    
                    selected_page_num = st.selectbox(
                        "Select page number:", 
                        page_options,
                        key="inspect_page_select"
                    )
                    
                    # 3. Get Image & Convert to RGBA (Crucial for drawing)
                    page = pdf.pages[selected_page_num - 1]
                    # Resolution 150 is usually good for viewing
                    im_obj = page.to_image(resolution=150)
                    
                    # Force conversion to RGBA to ensure colors/drawings work
                    pil_image = im_obj.original.convert("RGBA")
                    
                    # Use PIL Draw directly for maximum reliability
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(pil_image)
                    
                    # --- SCALING FACTOR ---
                    # Coordinates are in PDF points (72 DPI). Image is 150 DPI.
                    scale = 150 / 72 
                    
                    # --- 4. DRAW HEADER (RED BOX) ---
                    # Tabula: (top, left, bottom, right)
                    h_top, h_left, h_bottom, h_right = HEADER_AREA
                    # Scale coordinates
                    draw.rectangle(
                        [h_left * scale, h_top * scale, h_right * scale, h_bottom * scale],
                        outline="red", 
                        width=5
                    )
                    
                    # --- 5. DRAW TABLE AREA (BLUE BOX) ---
                    t_top, t_left, t_bottom, t_right = TABLE_AREA
                    draw.rectangle(
                        [t_left * scale, t_top * scale, t_right * scale, t_bottom * scale],
                        outline="blue", 
                        width=5
                    )
                    
                    # --- 6. DRAW COLUMNS (GREEN LINES) ---
                    for col_x in COLUMNS_X:
                        x_pixel = col_x * scale
                        y_start = t_top * scale
                        y_end = t_bottom * scale
                        draw.line(
                            [(x_pixel, y_start), (x_pixel, y_end)], 
                            fill="green", 
                            width=3
                        )
                    
                    # 7. Display Result
                    st.image(
                        pil_image, 
                        caption=f"Visual Inspection: {selected_inspect_file.name} (Page {selected_page_num})", 
                        use_container_width=True
                    )
                    
            except Exception as e:
                st.error(f"Error reading PDF for inspection: {e}")
    else:
        st.write("Upload a file above to enable inspection.")

# ==========================================
#            MAIN PROCESS LOGIC
# ==========================================

if uploaded_files:
    if st.button("Process Files", type="primary"):
        all_dfs_meta = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # 1. Get Header Info
                header_dfs = read_pdf(tmp_file_path, area=HEADER_AREA, pages='1', pandas_options={'header': None, 'dtype': str}, stream=True)
                
                if not header_dfs: continue
                
                header_dataframe = header_dfs[0]
                try:
                    periode_raw = header_dataframe.loc[header_dataframe[0] == 'PERIODE', 2].values[0]
                    periode = ' '.join(reversed(periode_raw.split()))
                    year_val, month_val = get_year_month(periode)
                except: year_val, month_val = 0, 0
                
                try: no_rekening = header_dataframe.loc[header_dataframe[0] == 'NO. REKENING', 2].values[0]
                except: no_rekening = "UNKNOWN"

                # 2. Get Transaction Data
                dataframes = read_pdf(tmp_file_path, area=TABLE_AREA, columns=COLUMNS_X, pages='all', pandas_options={'header': None, 'dtype': str}, force_subprocess=True)

                if dataframes:
                    try:
                        init_bal_row = dataframes[0].loc[dataframes[0][1] == 'SALDO AWAL', 5].values
                        init_balance = float(str(init_bal_row[0]).replace(',', '')) if len(init_bal_row) > 0 else 0.0

                        df = union_source(dataframes)
                        df = clean_numeric_columns(df, ['amount', 'balance'])
                        df = insert_shifted_column(df)
                        
                        t_df = extract_transactions(df)
                        if 'balance' in t_df.columns: t_df = t_df.drop('balance', axis=1)
                        t_df = calculate_balance(t_df, init_balance)

                        t_df['account_number'] = no_rekening
                        t_df = apply_categories(t_df)
                        t_df['transaction_type'] = t_df['transaction_type'].replace({'DB': 'DEBIT', 'CR': 'CREDIT'})

                        df_for_combine = t_df.copy()
                        df_for_combine['year'] = year_val
                        all_dfs_meta.append({'year': year_val, 'month_int': month_val, 'df': df_for_combine})
                        
                    except Exception as e: st.error(f"Error processing content in {uploaded_file.name}: {e}")

            except Exception as e: st.error(f"Failed to read PDF {uploaded_file.name}: {e}")
            finally: 
                if os.path.exists(tmp_file_path): os.unlink(tmp_file_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Done!")

        if all_dfs_meta:
            all_dfs_meta.sort(key=lambda x: (x['year'], x['month_int']))
            ordered_dfs = [x['df'] for x in all_dfs_meta]
            combined_df = pd.concat(ordered_dfs, ignore_index=True)

            def format_date_cols(row):
                try:
                    day, month = map(int, row['date'].split('/'))
                    dt = datetime(row['year'], month, day)
                    return pd.Series([dt.strftime('%Y-%m-%d'), dt.strftime('%b-%Y')])
                except: return pd.Series([row['date'], str(row['year'])])

            combined_df[['Date', 'Month']] = combined_df.apply(format_date_cols, axis=1)

            cols = ['Date', 'Month', 'bank_name', 'account_number', 'desc', 'category', 'beneficiary_name', 'amount', 'transaction_type', 'balance']
            final_df = combined_df[cols]

            st.session_state.processed_df = final_df
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False, sheet_name='Combined')
                worksheet = writer.sheets['Combined']
                currency_format = '_-Rp* #,##0.00_-;[Red]-Rp* #,##0.00_-;_-Rp* "-"_-;_-@_-'
                
                amount_col_idx = None
                balance_col_idx = None
                for cell in worksheet[1]:
                    if cell.value == 'amount': amount_col_idx = cell.column
                    elif cell.value == 'balance': balance_col_idx = cell.column
                
                if amount_col_idx:
                    for row in worksheet.iter_rows(min_row=2, min_col=amount_col_idx, max_col=amount_col_idx):
                        for cell in row: cell.number_format = currency_format
                
                if balance_col_idx:
                    for row in worksheet.iter_rows(min_row=2, min_col=balance_col_idx, max_col=balance_col_idx):
                        for cell in row: cell.number_format = currency_format

                for column_cells in worksheet.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter 
                    for cell in column_cells:
                        try:
                            if len(str(cell.value)) > max_length: max_length = len(str(cell.value))
                        except: pass
                    
                    adjusted_width = (max_length + 2)
                    if column_cells[0].column in [amount_col_idx, balance_col_idx]: adjusted_width += 10
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            st.session_state.excel_bytes = output.getvalue()
        else:
            st.warning("No transactions found.")

if st.session_state.processed_df is not None:
    st.success("Successfully processed data!")
    st.dataframe(
        st.session_state.processed_df,
        column_config={
            "amount": st.column_config.NumberColumn("Amount (IDR)", format="Rp %.2f"),
            "balance": st.column_config.NumberColumn("Balance (IDR)", format="Rp %.2f"),
        }
    )

    st.subheader("Actions")
    st.download_button(
        label="💾 Download as Excel",
        data=st.session_state.excel_bytes,
        file_name="combined_statement.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

    with st.expander("📋 Copy & Paste Data (for Sheets/Excel)"):
        st.info("Click the copy icon in the top-right corner of the black box below.")
        csv_text = st.session_state.processed_df.to_csv(sep='\t', index=False)
        st.code(csv_text, language="text")
