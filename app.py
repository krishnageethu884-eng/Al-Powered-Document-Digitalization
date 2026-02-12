import streamlit as st 
import pandas as pd
import digitalizer
import base64
import os
import concurrent.futures



st.set_page_config(page_title="School Document Digitalizer", layout="wide", page_icon="🏫")


DB_FILE = "output.csv"
REQUIRED_COLUMNS = ['filename', 'register_number', 'name', 'sex', 'dob', 'father_name', 'mother_name', 'address']


def load_db():
    if os.path.exists(DB_FILE):
        return pd.read_csv(DB_FILE)
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def save_db(df):
    current_db = load_db()
    updated_db = pd.concat([current_db, df], ignore_index=True)
    updated_db.to_csv(DB_FILE, index=False)
    return updated_db


st.sidebar.title("🏫 Digitalizer System")
page = st.sidebar.radio("Go to", ["Teacher's Portal", "Database Record"])

if page == "Teacher's Portal":
    st.title("👩‍🏫 Teacher's Portal - Data Entry")
    st.markdown("Upload Student Documents (Images or PDFs) to digitize them.")


    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'pdf'])

    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = pd.DataFrame(columns=REQUIRED_COLUMNS)

    if uploaded_files:
        if st.button(f"Process {len(uploaded_files)} Files"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            total_files = len(uploaded_files)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Prepare arguments for processing
                future_to_file = {
                    executor.submit(digitalizer.process_file_data, file.getvalue(), file.name): file 
                    for file in uploaded_files
                }
                
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        data = future.result()
                        results.extend(data)
                    except Exception as exc:
                        st.error(f"{file.name} generated an exception: {exc}")
                    
                    completed_count += 1
                    progress_bar.progress(completed_count / total_files)
                    status_text.text(f"Processed {completed_count}/{total_files} files...")
            
            status_text.text("Processing Complete!")
            
            if results:
                new_df = pd.DataFrame(results)
                
                for c in REQUIRED_COLUMNS:
                    if c not in new_df.columns:
                        new_df[c] = ""
                st.session_state.extracted_data = new_df[REQUIRED_COLUMNS]
            else:
                st.warning("No text extracted. Check file quality.")

    
    if not st.session_state.extracted_data.empty:
        st.subheader("📝 Verify & Edit Details")
        st.info("Please review the extracted data below. You can edit cells directly before saving.")
        
        edited_df = st.data_editor(st.session_state.extracted_data, num_rows="dynamic", use_container_width=True)
        
        
        if st.button("💾 Save to Database", type="primary"):
            save_db(edited_df)
            st.success(f"Saved {len(edited_df)} records to Database ({DB_FILE})!")
            st.session_state.extracted_data = pd.DataFrame(columns=REQUIRED_COLUMNS) # Clear after save

elif page == "Database Record":
    st.title("🗄️ Student Database Records")
    
    df = load_db()
    
    if not df.empty:
        
        search_query = st.text_input("🔍 Search (Name, Register No, etc.)")
        
        if search_query:
            mask = df.apply(lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)
            display_df = df[mask]
        else:
            display_df = df
            
        st.write(f"Showing {len(display_df)} records")
        
        st.dataframe(
            display_df,
            column_config={
                "address": st.column_config.TextColumn("Address", width="large"),
                "dob": st.column_config.DateColumn("Date of Birth", format="DD/MM/YYYY"),
            },
            use_container_width=True
        )

        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
             display_df.to_excel(writer, index=False)
        
        st.download_button(
            label="Download data as Excel",
            data=buffer,
            file_name='student_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("Database is empty. Go to Teacher's Portal to add records.")
