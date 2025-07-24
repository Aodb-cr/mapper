import streamlit as st
import subprocess
import json
import os
import pandas as pd
from pathlib import Path
from feedback_loop_system import FeedbackLoopManager, FeedbackRecord

# --- Configuration ---
PROJECT_DIR = Path(__file__).parent.resolve()
SMART_MAPPER_SCRIPT = PROJECT_DIR / "smart_mapper_main.py"
TEMP_DIR = PROJECT_DIR / "temp_uploads"
TEMP_DIR.mkdir(exist_ok=True)
feedback_manager = FeedbackLoopManager()


# --- Helper Functions ---

def get_schema_choices():
    return ["Auto-Detect", "cortex_xdr", "crowdstrike", "fortigate", "trellix_epo", "trend_micro", "generic_security"]

def get_target_fields():
    return sorted([
        'alert_name', 'description', 'incident_type', 'device_type', 'severity',
        'contexts.hostname', 'contexts.os', 'contexts.user', 'events.file_name',
        'events.file_path', 'events.cmd_line', 'events.hash.sha256', 'events.hash.sha1', 'events.hash.md5', 'events.src_ip',
        'events.src_port', 'events.dst_ip', 'events.dst_port', 'events.domain',
        'detected_time', 'log_source', 'rule_name', 'url', 'source_ip', 'destination_ip',
        'rawAlert'
    ])

def get_all_json_paths(data, parent_key='', sep='.'):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(get_all_json_paths(v, new_key, sep=sep))
            else:
                items.append(new_key)
    elif isinstance(data, list):
        if data:
            items.extend(get_all_json_paths(data[0], parent_key + '[0]', sep=sep))
    return sorted(list(set(items)))

def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        elif isinstance(x, list):
            out[name[:-1]] = json.dumps(x)
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def find_unmapped_fields(target_fields, mapped_data):
    unmapped = []
    flat_mapped_data = flatten_json(mapped_data)
    for field in target_fields:
        if field not in flat_mapped_data or flat_mapped_data[field] is None or str(flat_mapped_data[field]).strip() == '':
            unmapped.append(field)
    return unmapped

# NO CACHE on this function to ensure it ALWAYS runs with the latest knowledge
def run_mapper(_input_file_bytes, _uploaded_file_name, schema):
    temp_file_path = TEMP_DIR / _uploaded_file_name
    with open(temp_file_path, "wb") as f:
        f.write(_input_file_bytes)
    command = ["python", str(SMART_MAPPER_SCRIPT), "--input", str(temp_file_path)]
    if schema and schema != "Auto-Detect":
        command.extend(["--force-schema", schema])
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=120)
        stdout, stderr = process.stdout, process.stderr
        output_file_path = next((line.split("Full path:")[1].strip() for line in stdout.splitlines() if "Full path:" in line), None)
        if output_file_path and Path(output_file_path).exists():
            with open(output_file_path, 'r', encoding='utf-8') as f: result_data = json.load(f)
            quality_score = next((float(line.split(":")[1].replace('%','').strip()) for line in stdout.splitlines() if "Quality:" in line), 0.0)
            return {"success": True, "quality": quality_score, "output_data": result_data, "log": stdout + "\n" + stderr, "schema": schema}
        else:
            return {"success": False, "error": "Could not find or open the output file.", "log": stdout + "\n" + stderr}
    except Exception as e:
        return {"success": False, "error": f"An unexpected error occurred: {e}"}

def teach_pattern(target_field, source_path, schema, _input_file_bytes, _uploaded_file_name, repetitions=1):
    temp_file_path = TEMP_DIR / _uploaded_file_name
    with open(temp_file_path, "wb") as f:
        f.write(_input_file_bytes)
    command = ["python", str(SMART_MAPPER_SCRIPT), "--teach", target_field, source_path, schema, "--input", str(temp_file_path)]
    final_message = ""
    success = False
    for i in range(repetitions):
        try:
            process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=60)
            final_message += f"Run {i+1}/{repetitions}:\n{process.stdout}\n{process.stderr}\n---\n"
            if process.returncode == 0 and "Taught system" in process.stdout:
                success = True
        except Exception as e:
            return False, f"An error occurred during repetition {i+1}: {e}"

    if success:
        try:
            feedback_record = FeedbackRecord(
                field_name=target_field,
                schema_type=schema,
                corrected_source_path=source_path,
                user_action="corrected",
                file_source=_uploaded_file_name
            )
            feedback_manager.db.save_feedback(feedback_record)
            final_message += "\n✅ Feedback saved to database."
        except Exception as e:
            final_message += f"\n⚠️ Could not save feedback to database: {e}"

    return success, final_message

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Smart Data Mapper")
st.title("🧠 Smart Data Mapper V10 - The Reliable Overhaul")
st.markdown("A stable, two-step process: **1. Teach**, then **2. Re-process** to see new results.")

# --- Session State Initialization ---
if 'feedback_items' not in st.session_state: st.session_state.feedback_items = []
if 'source_paths' not in st.session_state: st.session_state.source_paths = []
if 'result' not in st.session_state: st.session_state.result = None
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload your JSON file", type=["json"])
    schema_choice = st.selectbox("Select Target Schema", options=get_schema_choices(), key="schema_choice")
    if st.button("Process File", type="primary", disabled=not uploaded_file):
        st.session_state.feedback_items = []
        st.session_state.feedback_submitted = False
        uploaded_file_bytes = uploaded_file.getvalue()
        with st.spinner("Processing file and extracting source paths..."):
            try:
                source_json = json.loads(uploaded_file_bytes)
                st.session_state.source_paths = get_all_json_paths(source_json)
                result = run_mapper(uploaded_file_bytes, uploaded_file.name, schema_choice)
                st.session_state.result = result
                st.session_state.uploaded_file_bytes = uploaded_file_bytes
                st.session_state.uploaded_file_name = uploaded_file.name
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please upload a valid JSON.")
                st.session_state.result = None

# --- Main Area ---
if st.session_state.result is None:
    st.info("Please upload a valid JSON file and click 'Process' to begin.")
else:
    result = st.session_state.result
    if not result["success"]:
        st.error(f"An error occurred: {result.get('error', 'Unknown error')}")
        st.code(result.get("log", "No log available."), language='bash')
    else:
        st.header("Processing Results")
        st.metric(label="Mapping Quality", value=f"{result['quality']:.2f}%")
        output_col, action_col = st.columns([3, 2])

        with output_col:
            st.subheader("Mapped Output (View Only)")
            with st.container(height=800):
                st.json(result["output_data"], expanded=True)

        with action_col:
            st.subheader("✍️ Step 1: Teach the System")
            with st.form("add_feedback_form", clear_on_submit=True):
                target_field = st.selectbox("Target Field", options=get_target_fields(), key="target")
                source_path = st.selectbox("Source Path (from uploaded file)", options=st.session_state.source_paths, key="source")
                if st.form_submit_button("Add Feedback to Batch") and target_field and source_path:
                    st.session_state.feedback_items.append({"target": target_field, "source": source_path})
            
            if st.session_state.feedback_items:
                st.write("**Feedback Batch:**")
                for item in st.session_state.feedback_items:
                    st.markdown(f"- **Map from:** `{item['source']}` <br>**To:** `{item['target']}`", unsafe_allow_html=True)
                
                teach_strength = st.slider("Teaching Strength", 1, 10, 3, help="Submits the feedback multiple times to make learning permanent.")
                
                if st.button("Submit Feedback", type="primary"):
                    schema_for_teaching = result.get("schema", "generic_security")
                    if schema_for_teaching == "Auto-Detect":
                        st.warning("Teaching requires a specific schema. Defaulting to 'generic_security'.")
                        schema_for_teaching = "generic_security"
                    
                    progress_bar = st.progress(0, text="Submitting feedback...")
                    total_items = len(st.session_state.feedback_items)
                    success_count = 0
                    for i, item in enumerate(st.session_state.feedback_items):
                        with st.spinner(f"Teaching item {i+1}/{total_items} with strength {teach_strength}..."):
                            success, msg = teach_pattern(item['target'], item['source'], schema_for_teaching, st.session_state.uploaded_file_bytes, st.session_state.uploaded_file_name, repetitions=teach_strength)
                            if success: success_count += 1
                        progress_bar.progress((i + 1) / total_items, text=f"Submitting... {i+1}/{total_items}")
                    
                    st.session_state.feedback_items = [] # Clear the batch
                    st.session_state.feedback_submitted = True # Set flag to show re-run button
                    st.success(f"✅ Teach complete! {success_count}/{total_items} items learned. Now click the button below to see the new results.")

            st.divider()

            # --- Step 2: Re-process ---
            st.subheader("Step 2: See New Results")
            if st.session_state.feedback_submitted:
                if st.button("🔄 Re-process File with New Knowledge", type="primary"):
                    with st.spinner("Re-processing with new knowledge..."):
                        new_result = run_mapper(
                            st.session_state.uploaded_file_bytes, 
                            st.session_state.uploaded_file_name, 
                            st.session_state.get('schema_choice')
                        )
                        st.session_state.result = new_result
                        st.session_state.feedback_submitted = False # Reset flag
                        st.toast("Results updated!", icon="✨")
            else:
                st.info("Submit feedback in Step 1 to enable this button.")

            st.divider()

            st.subheader("⚠️ Unmapped Fields")
            st.markdown("The following fields are missing or empty in the output.")
            unmapped = find_unmapped_fields(get_target_fields(), result['output_data'])
            if unmapped:
                with st.container(height=250):
                    for field in unmapped:
                        st.code(field, language='text')
            else:
                st.success("All target fields have been mapped!")

        with st.expander("View Full Processing Log"):
            st.code(result.get("log", "No log available."), language='bash')