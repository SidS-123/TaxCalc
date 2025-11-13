import streamlit as st
import pandas as pd
import re

# ------------------ Tax Config (easily editable each year) ------------------
TAX_RATES = {
    "ss_rate": 0.062,
    "ss_wage_limit": 168600,         # 2025 limit
    "medicare_rate": 0.0145,
    "addl_medicare_rate": 0.009,
    "addl_medicare_threshold": 200000,
    "fed_est_rate": 0.12,            # simple placeholder for demo
    "dep_care_limit": 5000,
    "401k_limit": 23000,
}

# ------------------ Streamlit Layout ------------------
st.set_page_config(page_title="TaxAI - W-2 Calculator", layout="wide")
st.title("ðŸ§¾ TaxAI: W-2 Tax Info Calculator (2025 Edition)")

st.markdown("""
Upload a **W-2 CSV file** â€” the app will automatically detect column names, 
compute derived federal, Social Security, and Medicare tax data, and 
apply current-year IRS rates.
""")

uploaded_file = st.file_uploader("ðŸ“‚ Upload W-2 CSV", type=["csv"])

# ------------------ Helper Functions ------------------
def find_col(df, patterns):
    """Fuzzy find a column in df using a list of regex patterns."""
    for pattern in patterns:
        for col in df.columns:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None


def get(df, mappings, key):
    """Safely return df[column] or 0 if not found."""
    colname = mappings.get(key)
    return df[colname] if colname and colname in df.columns else 0


# ------------------ Main Logic ------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # normalize headers to avoid invisible mismatches (BOMs, whitespace)
    df.columns = [str(c).strip().replace('\ufeff', '') for c in df.columns]
    st.subheader("ðŸ“‹ Raw W-2 Data")
    ### CANONICAL_W2_FIELDS_INSERTED
    # Normalize and ensure canonical W-2 fields (added by automation)
    try:
        required_cols = ['year','employer_name','employer_id','employee_name','ssn','wages','taxable_wages','federal_income_tax_withheld','social_security_wages','social_security_tax_withheld','medicare_wages','medicare_tax_withheld','retirement_deferrals','dependent_care_benefits','state','state_wages','state_tax_withheld','local_wages','local_tax_withheld','locality_name','net_pay','social_security_tax_rate','medicare_tax_rate']
        # normalize column names to lower-case strings
        df.columns = [str(c).strip().lower() for c in df.columns]
        num_cols = ['year','wages','taxable_wages','federal_income_tax_withheld','social_security_wages','social_security_tax_withheld','medicare_wages','medicare_tax_withheld','retirement_deferrals','dependent_care_benefits','state_wages','state_tax_withheld','local_wages','local_tax_withheld','net_pay','social_security_tax_rate','medicare_tax_rate']
        for c in required_cols:
            if c not in df.columns:
                if c in num_cols:
                    df[c] = 0
                else:
                    df[c] = ''
        # coerce numeric-like columns to numeric safely
        for c in num_cols:
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            except Exception:
                pass
    except Exception:
        # leave df unchanged on unexpected errors
        pass

    st.dataframe(df, use_container_width=True)

    # --- Flexible column mapping ---
    mappings = {
        "Employee": find_col(df, [r"employee", r"name"]),
        # employer and identity fields
        "EmployerName": find_col(df, [r"employer", r"company", r"employer_name"]),
        "EmployerID": find_col(df, [r"ein", r"employer.*id", r"employer_id"]),
        "SSN": find_col(df, [r"\bssn\b", r"social.*security", r"ssn"]),
        "Box1_Wages": find_col(df, [r"box\s*1", r"wages", r"wage\s*income"]),
        "Box2_FedWithheld": find_col(df, [r"box\s*2", r"federal.*withheld", r"fed.*tax"]),
        "Box3_SSWages": find_col(df, [r"box\s*3", r"ss.*wages", r"social.*security.*wages"]),
        "Box4_SSTaxWithheld": find_col(df, [r"box\s*4", r"ss.*tax.*withheld", r"social.*security.*tax"]),
        "Box5_MedicareWages": find_col(df, [r"box\s*5", r"medicare.*wages"]),
        # Medicare withheld can be labeled in many ways; try a wide set of patterns
        "Box6_MedicareWithheld": find_col(
            df,
            [
                r"box\s*6",
                r"medicare.*withheld",
                r"medicare.*tax",
                r"medicaretax",
                r"medicarewithheld",
                r"medicare",
            ],
        ),
        "Box10_DependentCare": find_col(df, [r"box\s*10", r"dependent.*care"]),
        "Box12_CodeD": find_col(df, [r"box\s*12", r"401k", r"retirement"]),
        "Box16_StateWages": find_col(df, [r"box\s*16", r"state.*wages"]),
        "Box17_StateWithheld": find_col(df, [r"box\s*17", r"state.*tax.*withheld"]),
        # local tax fields often appear in boxes 18-20 or labelled local
        "Box18_LocalWages": find_col(df, [r"box\s*18", r"local.*wages"]),
        "Box19_LocalWithheld": find_col(df, [r"box\s*19", r"local.*tax.*withheld", r"local.*withheld"]),
        "Box20_LocalityName": find_col(df, [r"box\s*20", r"locality", r"locality_name"]),
        "Year": find_col(df, [r"year", r"tax\s*year"]),
    }

    missing = [k for k, v in mappings.items() if v is None]
    if missing:
        st.warning(f"âš ï¸ Missing or unrecognized columns for: {', '.join(missing)}")

    # --- Core Calculations ---
    df_result = pd.DataFrame()
    df_result["Employee"] = get(df, mappings, "Employee")
    df_result["Total_Fed_Income"] = get(df, mappings, "Box1_Wages")
    df_result["Total_Fed_Withheld"] = get(df, mappings, "Box2_FedWithheld")

    # Social Security (cap at wage limit)
    ss_wages = get(df, mappings, "Box3_SSWages").clip(upper=TAX_RATES["ss_wage_limit"])
    df_result["Expected_SS_Tax"] = ss_wages * TAX_RATES["ss_rate"]
    df_result["SS_Tax_Diff"] = get(df, mappings, "Box4_SSTaxWithheld") - df_result["Expected_SS_Tax"]

    # Medicare base tax
    medicare_wages = get(df, mappings, "Box5_MedicareWages")
    df_result["Expected_Medicare_Tax"] = medicare_wages * TAX_RATES["medicare_rate"]

    # Additional Medicare (over threshold)
    df_result["Additional_Medicare_Tax"] = (
        (medicare_wages - TAX_RATES["addl_medicare_threshold"]).clip(lower=0)
        * TAX_RATES["addl_medicare_rate"]
    )

    df_result["Total_Medicare_Tax_Liability"] = (
        df_result["Expected_Medicare_Tax"] + df_result["Additional_Medicare_Tax"]
    )
    df_result["Medicare_Tax_Diff"] = get(df, mappings, "Box6_MedicareWithheld") - df_result["Total_Medicare_Tax_Liability"]

    # 401(k) and Dependent Care
    df_result["Retirement_Contrib_401k"] = get(df, mappings, "Box12_CodeD")
    df_result["Over_401k_Limit"] = df_result["Retirement_Contrib_401k"] > TAX_RATES["401k_limit"]
    df_result["Dependent_Care_Benefit"] = get(df, mappings, "Box10_DependentCare")
    df_result["Dep_Care_Over_Limit"] = df_result["Dependent_Care_Benefit"] > TAX_RATES["dep_care_limit"]

    # State
    df_result["State_Taxable_Income"] = get(df, mappings, "Box16_StateWages")
    df_result["State_Tax_Withheld"] = get(df, mappings, "Box17_StateWithheld")

    # Simple refund/owed estimate (rough)
    df_result["Estimated_Fed_Liability"] = get(df, mappings, "Box1_Wages") * TAX_RATES["fed_est_rate"]
    df_result["Potential_Refund_or_Owed"] = (
        get(df, mappings, "Box2_FedWithheld") - df_result["Estimated_Fed_Liability"]
    )

    # --- Display Results ---
    st.subheader("ðŸ“Š Derived Tax Calculations (2025 Rates)")
    # Render boolean-like columns as explicit 'true' / 'false' strings
    df_display = df_result.copy()
    # Sanitize display DataFrame: avoid showing literal None/"None"/"NONE" in the UI
    try:
        df_display = df_display.fillna("")
        df_display = df_display.replace({None: "", "None": "", "NONE": ""})
    except Exception:
        pass
    def is_bool_like(s: pd.Series) -> bool:
        # True if dtype is boolean or values are only boolean-like (0/1/true/false/yes/no)
        if pd.api.types.is_bool_dtype(s):
            return True
        vals = s.dropna().unique()
        if len(vals) == 0:
            return False
        lowered = {str(v).strip().lower() for v in vals}
        bool_like_set = {"true", "false", "1", "0", "yes", "no", "y", "n"}
        return lowered.issubset(bool_like_set)

    for c in df_display.columns:
        try:
            if is_bool_like(df_display[c]):
                df_display[c] = df_display[c].apply(lambda v: 'true' if str(v).strip().lower() in ('true', '1', 'yes', 'y') else 'false')
        except Exception:
            # if any unexpected error, skip conversion for that column
            pass

    st.dataframe(df_display, use_container_width=True)

    # --- Human-readable list of derived calculations ---
    st.markdown("### Derived fields explained")
    derived_descriptions = {
        "Expected_SS_Tax": "Social Security tax expected = min(SS wages, wage limit) Ã— 6.2%",
        "SS_Tax_Diff": "Difference between reported Social Security tax withheld and expected SS tax",
        "Expected_Medicare_Tax": "Medicare tax expected = Medicare wages Ã— 1.45%",
        "Additional_Medicare_Tax": "Additional Medicare tax on wages above threshold (0.9% over $200,000)",
        "Total_Medicare_Tax_Liability": "Sum of expected Medicare tax + additional Medicare tax",
        "Medicare_Tax_Diff": "Difference between reported Medicare tax withheld and total Medicare liability",
        "Retirement_Contrib_401k": "Reported 401(k) contributions (Box 12 code D if present)",
        "Over_401k_Limit": "Boolean: contribution exceeds the annual 401(k) limit",
        "Dependent_Care_Benefit": "Reported dependent care benefits (Box 10)",
        "Dep_Care_Over_Limit": "Boolean: dependent care benefit exceeds the annual limit",
        "Estimated_Fed_Liability": "Simple estimated federal liability used for a rough refund estimate (placeholder rate)",
        "Potential_Refund_or_Owed": "Approximate difference: federal tax withheld âˆ’ estimated federal liability",
    }

    # Render as a neat markdown table
    md_lines = ["| Field | Description |", "|---|---|"]
    for k, v in derived_descriptions.items():
        md_lines.append(f"| `{k}` | {v} |")
    st.markdown("\n".join(md_lines))

    # --- Produce final CSV with requested schema ---
    out_cols = [
        "year",
        "employer_name",
        "employer_id",
        "employee_name",
        "ssn",
        "wages",
        "taxable_wages",
        "federal_income_tax_withheld",
        "social_security_wages",
        "social_security_tax_withheld",
        "medicare_wages",
        "medicare_tax_withheld",
        "retirement_deferrals",
    "dependent_care_benefits",
        "state",
        "state_wages",
        "state_tax_withheld",
        "local_wages",
        "local_tax_withheld",
        "locality_name",
        "net_pay",
        "social_security_tax_rate",
        "medicare_tax_rate",
        # include some derived fields so the export contains the calculated tax checks
        "Total_Fed_Income",
        "Total_Fed_Withheld",
        "Expected_SS_Tax",
        "SS_Tax_Diff",
        "Expected_Medicare_Tax",
        "Additional_Medicare_Tax",
        "Total_Medicare_Tax_Liability",
        "Medicare_Tax_Diff",
        "Retirement_Contrib_401k",
    ]

    # helper to safely pull series or scalar
    def pull(col_key, fallback=0):
        try:
            return get(df, mappings, col_key)
        except Exception:
            return fallback

    final = pd.DataFrame()
    # prefer pulling values from uploaded CSV when available
    try:
        year_val = pd.to_numeric(pull("Year", 2025), errors="coerce")
        year_val = int(year_val) if pd.notnull(year_val) else 2025
    except Exception:
        year_val = 2025
    final["year"] = year_val
    final["employer_name"] = pull("EmployerName", "")
    final["employer_id"] = pull("EmployerID", "")
    final["employee_name"] = df_result.get("Employee") if "Employee" in df_result else pull("Employee")
    final["ssn"] = pull("SSN", "")
    # numeric pulls (coerce to numeric)
    final["wages"] = pd.to_numeric(df_result.get("Total_Fed_Income") if "Total_Fed_Income" in df_result else pull("Box1_Wages"), errors="coerce")
    final["taxable_wages"] = final["wages"]
    final["federal_income_tax_withheld"] = pd.to_numeric(pull("Box2_FedWithheld"), errors="coerce")
    final["social_security_wages"] = pd.to_numeric(pull("Box3_SSWages"), errors="coerce")
    final["social_security_tax_withheld"] = pd.to_numeric(pull("Box4_SSTaxWithheld"), errors="coerce")
    final["medicare_wages"] = pd.to_numeric(pull("Box5_MedicareWages"), errors="coerce")
    final["medicare_tax_withheld"] = pd.to_numeric(pull("Box6_MedicareWithheld"), errors="coerce")
    final["retirement_deferrals"] = pd.to_numeric(pull("Box12_CodeD"), errors="coerce")
    final["dependent_care_benefits"] = pd.to_numeric(pull("Box10_DependentCare"), errors="coerce")
    final["state"] = ""
    final["state_wages"] = pd.to_numeric(pull("Box16_StateWages"), errors="coerce")
    final["state_tax_withheld"] = pd.to_numeric(pull("Box17_StateWithheld"), errors="coerce")
    # local fields: prefer uploaded values when present
    final["local_wages"] = pd.to_numeric(pull("Box18_LocalWages", ""), errors="coerce")
    final["local_tax_withheld"] = pd.to_numeric(pull("Box19_LocalWithheld", ""), errors="coerce")
    final["locality_name"] = pull("Box20_LocalityName", "")
    # net_pay = wages - federal - ss - medicare (best-effort)
    final["net_pay"] = final["wages"] - final["federal_income_tax_withheld"].fillna(0) - final["social_security_tax_withheld"].fillna(0) - final["medicare_tax_withheld"].fillna(0)
    # Format rate columns as strings with sensible precision so CSV preserves values like 0.062 and 0.0145
    final["social_security_tax_rate"] = ('{:.4f}'.format(TAX_RATES["ss_rate"]).rstrip('0').rstrip('.'))
    final["medicare_tax_rate"] = ('{:.4f}'.format(TAX_RATES["medicare_rate"]).rstrip('0').rstrip('.'))

    # Add derived/calculated fields from df_result so the export CSV contains them too
    try:
        final["Total_Fed_Income"] = pd.to_numeric(df_result.get("Total_Fed_Income"), errors="coerce")
        final["Total_Fed_Withheld"] = pd.to_numeric(df_result.get("Total_Fed_Withheld"), errors="coerce")
        final["Expected_SS_Tax"] = pd.to_numeric(df_result.get("Expected_SS_Tax"), errors="coerce")
        final["SS_Tax_Diff"] = pd.to_numeric(df_result.get("SS_Tax_Diff"), errors="coerce")
        final["Expected_Medicare_Tax"] = pd.to_numeric(df_result.get("Expected_Medicare_Tax"), errors="coerce")
        final["Additional_Medicare_Tax"] = pd.to_numeric(df_result.get("Additional_Medicare_Tax"), errors="coerce")
        final["Total_Medicare_Tax_Liability"] = pd.to_numeric(df_result.get("Total_Medicare_Tax_Liability"), errors="coerce")
        final["Medicare_Tax_Diff"] = pd.to_numeric(df_result.get("Medicare_Tax_Diff"), errors="coerce")
        final["Retirement_Contrib_401k"] = pd.to_numeric(df_result.get("Retirement_Contrib_401k"), errors="coerce")
    except Exception:
        # non-fatal: leave missing values if df_result not available
        pass

    # Ensure column order
    final = final[out_cols]

    # Sanitize final output more carefully:
    # - For object/string columns: replace None/'None'/'NONE' with empty string and fill NaNs
    # - For numeric columns: fill NaNs with 0 so exported CSV contains numeric values instead of blanks
    try:
        obj_cols = final.select_dtypes(include=["object"]).columns.tolist()
        num_cols = final.select_dtypes(include=["number"]).columns.tolist()
        if obj_cols:
            final.loc[:, obj_cols] = final.loc[:, obj_cols].fillna("")
            final.loc[:, obj_cols] = final.loc[:, obj_cols].replace({None: "", "None": "", "NONE": ""})
        if num_cols:
            final.loc[:, num_cols] = final.loc[:, num_cols].fillna(0)
    except Exception:
        # non-fatal sanitization error; proceed without aborting the app
        pass

    st.subheader("ðŸ“¥ Export-ready CSV (standard schema)")
    st.dataframe(final, use_container_width=True)

    # Export with per-column formatting: round monetary values to 2 decimals
    # but keep tax-rate columns with higher precision (4 decimals) so 0.062 and 0.0145 are preserved.
    try:
        export_df = final.copy()
        num_cols = export_df.select_dtypes(include=["number"]).columns.tolist()
        # round monetary/numeric columns to 2 decimals by default
        if num_cols:
            export_df.loc[:, num_cols] = export_df.loc[:, num_cols].round(2)
        # keep tax rate precision
        rate_cols = ["social_security_tax_rate", "medicare_tax_rate"]
        # Format rate columns as strings with up to 4 decimal places but trim trailing zeros
        for rc in rate_cols:
            if rc in export_df.columns:
                try:
                    export_df[rc] = export_df[rc].apply(lambda v: ("{:.4f}".format(v)).rstrip('0').rstrip('.') if pd.notnull(v) else "")
                except Exception:
                    # fallback to numeric rounding if apply fails
                    export_df[rc] = export_df[rc].round(4)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    except Exception:
        # fallback to simple CSV if something goes wrong
        csv_bytes = final.to_csv(index=False, float_format='%.2f').encode("utf-8")
    st.download_button("Download standard CSV", data=csv_bytes, file_name="parsed_w2_standard.csv", mime="text/csv")

    # Download CSV
    # Export derived results with sensible rounding
    try:
        dr = df_result.copy()
        num_cols = dr.select_dtypes(include=["number"]).columns.tolist()
        if num_cols:
            dr.loc[:, num_cols] = dr.loc[:, num_cols].round(2)
        csv_out = dr.to_csv(index=False).encode("utf-8")
    except Exception:
        csv_out = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="ðŸ’¾ Download Derived Results",
        data=csv_out,
        file_name="derived_w2_taxinfo_2025.csv",
        mime="text/csv",
    )

    st.success("âœ… Tax data successfully processed!")
else:
    st.info("ðŸ‘† Upload a W-2 CSV to begin.")


