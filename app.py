# app.py
import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
from typing import Dict, Any

# ---------------------------
# COLUMN MAPPING (CSV header -> internal field)
# ---------------------------
COLUMN_MAP = {
    # Identification / metadata
    "profile_id": "profile_id",
    "year": "year",
    "employer_name": "employer_name",
    "employer_id": "employer_id",
    "employee_name": "employee_name",
    "ssn": "ssn",
    "state": "state",
    "locality_name": "locality_name",

    # Core wage fields
    "wages": "wages",
    "taxable_wages": "taxable_wages",

    # Federal withholding
    "federal_income_tax_withheld": "federal_withheld",

    # Social Security
    "social_security_wages": "ss_wages",
    "social_security_tax_withheld": "ss_withheld",

    # Medicare
    "medicare_wages": "med_wages",
    "medicare_tax_withheld": "med_withheld",

    # State & Local
    "state_wages": "state_wages",
    "state_tax_withheld": "state_tax",
    "local_wages": "local_wages",
    "local_tax_withheld": "local_tax",

    # Benefits (pretax contributions)
    "retirement_deferrals": "retirement_deferrals",         # 401k/403b/457
    "dependent_care_benefits": "dependent_care_benefits",   # DCB
}

# ---------------------------
# Tax tables (multi-year support)
# ---------------------------
IRS_BRACKETS = {
    2024: [
        (0, 11600, 0.10),
        (11600, 47150, 0.12),
        (47150, 100525, 0.22),
        (100525, 191950, 0.24),
        (191950, 243725, 0.32),
        (243725, 609350, 0.35),
        (609350, float("inf"), 0.37),
    ],
    2023: [
        (0, 11000, 0.10),
        (11000, 44725, 0.12),
        (44725, 95375, 0.22),
        (95375, 182100, 0.24),
        (182100, 231250, 0.32),
        (231250, 578125, 0.35),
        (578125, float("inf"), 0.37),
    ],
    2022: [
        (0, 10275, 0.10),
        (10275, 41775, 0.12),
        (41775, 89075, 0.22),
        (89075, 170050, 0.24),
        (170050, 215950, 0.32),
        (215950, 539900, 0.35),
        (539900, float("inf"), 0.37),
    ],
    2021: [
        (0, 9950, 0.10),
        (9950, 40525, 0.12),
        (40525, 86375, 0.22),
        (86375, 164925, 0.24),
        (164925, 209425, 0.32),
        (209425, 523600, 0.35),
        (523600, float("inf"), 0.37),
    ],
    2025: [  # placeholder
        (0, 12000, 0.10),
        (12000, 49000, 0.12),
        (49000, 105000, 0.22),
        (105000, 200000, 0.24),
        (200000, 270000, 0.32),
        (270000, 630000, 0.35),
        (630000, float("inf"), 0.37),
    ],
}

STANDARD_DEDUCTION = {
    2024: {"single": 14600, "married": 29200, "hoh": 21900},
    2023: {"single": 13850, "married": 27700, "hoh": 20700},
    2022: {"single": 12950, "married": 25900, "hoh": 19400},
    2021: {"single": 12550, "married": 25100, "hoh": 18800},
    2025: {"single": 15750, "married": 31500, "hoh": 23625},
}

SS_WAGE_BASE = {2021: 142800, 2022: 147000, 2023: 160200, 2024: 168600, 2025: 176100}
SS_RATE = 0.062
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_THRESHOLDS = {
    "single": 200000,
    "married": 250000,
    "married_separate": 125000,
    "hoh": 200000,
}

# ---------------------------
# Utilities
# ---------------------------
def to_number(x) -> float:
    """Attempt to convert common numeric strings to float. Return 0.0 on failure."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    # remove $ and commas and parentheses
    s = s.replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
    try:
        return float(s)
    except Exception:
        return 0.0


def normalize_row_for_internal(row: pd.Series) -> Dict[str, Any]:
    """
    Build an internal dict keyed by internal names using COLUMN_MAP.
    Non-mapped columns are left accessible via 'extra_*' keys.
    """
    normalized = {}
    # map known CSV headers
    for csv_key, internal_key in COLUMN_MAP.items():
        # allow case-insensitive match of headers
        matched_val = None
        for candidate in (csv_key, csv_key.upper(), csv_key.lower(), csv_key.title()):
            if candidate in row.index:
                matched_val = row[candidate]
                break
        if matched_val is None and csv_key in row.index:
            matched_val = row[csv_key]
        # convert to numeric where appropriate
        if matched_val is None:
            normalized[internal_key] = 0.0
        else:
            # for SSN or textual fields we keep as string if non-numeric
            if internal_key in ("employee_name", "employer_name", "ssn", "employer_id", "profile_id", "locality_name", "state"):
                normalized[internal_key] = matched_val
            else:
                normalized[internal_key] = to_number(matched_val)

    # carry-through any other raw fields as extra_*
    for col in row.index:
        if col not in COLUMN_MAP:
            normalized[f"extra_{col}"] = row[col]

    return normalized

# ---------------------------
# Data model classes
# ---------------------------
class W2Record:
    def __init__(self, data: Dict[str, Any]):
        """
        Accepts a dict containing internal field names (as produced by normalize_row_for_internal).
        """
        self.raw = data
        # string / id fields
        self.profile_id = data.get("profile_id", "")
        self.year = int(to_number(data.get("year", 0))) if data.get("year", "") != "" else None
        self.employer_name = data.get("employer_name", "")
        self.employer_id = data.get("employer_id", "")
        self.employee_name = data.get("employee_name", "") or data.get("EmployeeName", "")
        self.ssn = data.get("ssn", "")
        self.locality_name = data.get("locality_name", "")
        self.state = data.get("state", "")

        # numeric wage + tax fields
        self.wages = to_number(data.get("wages", 0.0))
        self.taxable_wages = to_number(data.get("taxable_wages", 0.0))
        self.federal_withheld = to_number(data.get("federal_withheld", 0.0))
        self.ss_wages = to_number(data.get("ss_wages", 0.0))
        self.ss_withheld = to_number(data.get("ss_withheld", 0.0))
        self.med_wages = to_number(data.get("med_wages", 0.0))
        self.med_withheld = to_number(data.get("med_withheld", 0.0))
        self.state_wages = to_number(data.get("state_wages", 0.0))
        self.state_tax = to_number(data.get("state_tax", 0.0))
        self.local_wages = to_number(data.get("local_wages", 0.0))
        self.local_tax = to_number(data.get("local_tax", 0.0))

        # benefits
        self.retirement_deferrals = to_number(data.get("retirement_deferrals", 0.0))
        self.dependent_care_benefits = to_number(data.get("dependent_care_benefits", 0.0))

        # derived/metadata defaults
        self.filing_status = data.get("filing_status", data.get("filingStatus", "single")).lower()

# ---------------------------
# Federal tax engine
# ---------------------------
class IRSFederalTaxCalculator:
    def __init__(self, taxable_income: float, year: int = 2024, filing_status: str = "single"):
        self.income = max(0.0, float(taxable_income))
        self.year = year if year in IRS_BRACKETS else 2024
        self.filing_status = filing_status if filing_status in STANDARD_DEDUCTION[self.year] else "single"
        self.brackets = IRS_BRACKETS[self.year]
        self.standard_deduction = STANDARD_DEDUCTION[self.year][self.filing_status]

    def compute_federal_tax(self):
        taxable_after_std = max(0.0, self.income - self.standard_deduction)
        tax = 0.0
        for lower, upper, rate in self.brackets:
            if taxable_after_std > lower:
                taxed_amount = min(taxable_after_std, upper) - lower
                if taxed_amount > 0:
                    tax += taxed_amount * rate
            else:
                break
        return tax, taxable_after_std

# ---------------------------
# Derived fields calculator
# ---------------------------
class W2DerivedCalculator:
    def __init__(self, rec: W2Record, tax_year: int = 2024):
        self.r = rec
        self.tax_year = tax_year
        self.ss_wage_cap = SS_WAGE_BASE.get(tax_year, SS_WAGE_BASE[2024])

    def compute_all(self) -> Dict[str, Any]:
        r = self.r
        derived = {}

        # identity
        derived["Employee"] = r.employee_name or r.profile_id or "Unknown"

        # basic
        derived["Annual Taxable Income (Box 1 / wages)"] = r.wages
        derived["Taxable Wages (taxable_wages)"] = r.taxable_wages

        # gross estimation
        gross_est = max(r.wages, r.ss_wages, r.med_wages, r.taxable_wages)
        derived["Total Gross Earnings (estimate)"] = gross_est

        # pretax inference
        derived["Inferred Pretax Reductions"] = gross_est - r.wages

        # Social Security
        ss_owed = SS_RATE * min(r.ss_wages, self.ss_wage_cap)
        derived["Social Security Tax Owed (expected)"] = ss_owed
        derived["Social Security Tax Withheld (Box 4)"] = r.ss_withheld
        derived["SS Withheld - Expected"] = r.ss_withheld - ss_owed
        derived["SS Wage Cap"] = self.ss_wage_cap
        derived["SS Wage Cap Exceeded"] = r.ss_wages > self.ss_wage_cap

        # Medicare
        med_basic = MEDICARE_RATE * r.med_wages
        threshold = ADDITIONAL_MEDICARE_THRESHOLDS.get(r.filing_status, ADDITIONAL_MEDICARE_THRESHOLDS["single"])
        extra_med = 0.0
        if r.med_wages > threshold:
            extra_med = (r.med_wages - threshold) * ADDITIONAL_MEDICARE_RATE
        med_total = med_basic + extra_med
        derived["Medicare Tax Owed (expected)"] = med_total
        derived["Medicare Tax Withheld (Box 6)"] = r.med_withheld
        derived["Medicare Withheld - Expected"] = r.med_withheld - med_total
        derived["Additional Medicare Threshold (filing status)"] = threshold

        # payroll totals & rates
        derived["Federal Tax Withheld (Box 2)"] = r.federal_withheld
        derived["Total Payroll Taxes Withheld (Fed+SS+Med)"] = r.federal_withheld + r.ss_withheld + r.med_withheld
        derived["Federal Withholding Rate (Box2/Box1)"] = (r.federal_withheld / r.wages) if r.wages else 0.0
        derived["State Tax Withheld"] = r.state_tax
        derived["Local Tax Withheld"] = r.local_tax

        # benefits
        derived["Retirement Deferrals (reported)"] = r.retirement_deferrals
        derived["Dependent Care Benefits (Box10)"] = r.dependent_care_benefits

        # EITC-like metric
        derived["Earned Income Metric (wages + imputed)"] = r.wages + 0.0  # imputed income not present in this CSV

        # error detection
        errors = []
        # negative checks
        numeric_fields = ["wages", "taxable_wages", "federal_withheld", "ss_wages", "ss_withheld", "med_wages", "med_withheld"]
        for f in numeric_fields:
            val = getattr(r, f, 0.0)
            if val < 0:
                errors.append(f"Negative value detected in {f}: {val}")

        if abs(r.ss_withheld - ss_owed) > 1.0:
            errors.append(f"SS withheld ({r.ss_withheld}) differs from expected ({ss_owed:.2f})")

        if abs(r.med_withheld - med_total) > 1.0:
            errors.append(f"Medicare withheld ({r.med_withheld}) differs from expected ({med_total:.2f})")

        # Box rule check
        if not (r.wages <= r.ss_wages + 1e-9 or r.wages <= r.med_wages + 1e-9):
            errors.append("Wages (Box1) exceed SS/Medicare wages (unusual)")

        derived["Detected Errors/Warnings"] = errors

        # federal tax via bracket engine
        fed_calc = IRSFederalTaxCalculator(r.wages, year=self.tax_year, filing_status=r.filing_status)
        fed_tax, taxable_after_std = fed_calc.compute_federal_tax()
        derived["Taxable Income After Standard Deduction"] = taxable_after_std
        derived["Federal Tax (computed via brackets)"] = fed_tax
        derived["Federal Refund / (Balance Due)"] = r.federal_withheld - fed_tax

        # consistency score
        score = 1.0
        if errors:
            score -= 0.2 * min(len(errors), 4)
        if r.ss_wages > self.ss_wage_cap:
            score -= 0.1
        derived["Consistency Score (0-1)"] = max(0.0, score)

        return derived

# ---------------------------
# Streamlit UI (Single Page)
# ---------------------------
st.set_page_config(page_title="W-2 Dashboard (Single Page)", layout="wide")
st.title("📄 W-2 Derived Fields Dashboard — Single Page")

st.markdown("Upload a CSV with W-2 rows (headers matched to your provided format). Select a tax year (default 2024).")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded = st.file_uploader("Upload W-2 CSV", type=["csv"])
with col2:
    tax_year = st.selectbox("IRS Tax Year", options=[2024, 2023, 2022, 2021, 2025], index=0)
    default_filing = st.selectbox("Default filing status", options=["single", "married", "hoh", "married_separate"], index=0)

if uploaded:
    df_raw = pd.read_csv(uploaded, dtype=str)  # read as strings to preserve formatting for parsing
    st.success(f"Loaded {len(df_raw)} rows from CSV")

    # Show parsed header preview
    with st.expander("CSV headers (preview)"):
        st.write(list(df_raw.columns))

    derived_rows = []
    info_rows = []
    # Keys to include in the 'Info' export (preserve original input/source fields)
    info_keys = [
        "profile_id",
        "year",
        "employer_name",
        "employer_id",
        "employee_name",
        "ssn",
        "wages",
        "taxable_wages",
        "state",
        "locality_name",
        "state_wages",
        "state_tax_withheld",
        "local_wages",
        "local_tax_withheld",
    ]
    for idx, row in df_raw.iterrows():
        normalized = normalize_row_for_internal(row)
        # ensure filing status
        if not normalized.get("filing_status"):
            normalized["filing_status"] = default_filing
        rec = W2Record(normalized)
        calc = W2DerivedCalculator(rec, tax_year=tax_year)
        derived = calc.compute_all()
        # capture original/input info fields for separate export/display
        info = {k: normalized.get(k) for k in info_keys}
        info_rows.append(info)
        derived_rows.append(derived)

    # Display results
    st.header("Derived Results")
    for i, d in enumerate(derived_rows):
        with st.expander(f"{d.get('Employee','Employee')} — Row {i+1}", expanded=(i == 0)):
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Key Summary")
                summary_keys = [
                    "Annual Taxable Income (Box 1 / wages)",
                    "Taxable Wages (taxable_wages)",
                    "Total Gross Earnings (estimate)",
                    "Inferred Pretax Reductions",
                    "Retirement Deferrals (reported)",
                    "Dependent Care Benefits (Box10)",
                    "Social Security Tax Owed (expected)",
                    "Social Security Tax Withheld (Box 4)",
                    "SS Withheld - Expected",
                    "Medicare Tax Owed (expected)",
                    "Medicare Tax Withheld (Box 6)",
                    "Medicare Withheld - Expected",
                    "Federal Tax (computed via brackets)",
                    "Federal Tax Withheld (Box 2)",
                    "Federal Refund / (Balance Due)",
                    "Total Payroll Taxes Withheld (Fed+SS+Med)",
                    "Consistency Score (0-1)",
                ]
                for k in summary_keys:
                    if k in d:
                        v = d[k]
                        if isinstance(v, (int, float)):
                            st.write(f"**{k}:** {v:,.2f}")
                        else:
                            st.write(f"**{k}:** {v}")

                st.markdown("---")
                st.subheader("Other Derived Fields")
                for k, v in d.items():
                    if k not in summary_keys and k not in ("Employee", "Detected Errors/Warnings"):
                        if isinstance(v, (int, float)):
                            st.write(f"**{k}:** {v:,.2f}")
                        else:
                            st.write(f"**{k}:** {v}")

            with right:
                st.subheader("Errors & Warnings")
                errs = d.get("Detected Errors/Warnings", [])
                if errs:
                    for e in errs:
                        st.error(e)
                else:
                    st.success("No immediate errors detected")

                # Charts
                st.subheader("Charts")
                # Payroll pie
                fig1, ax1 = plt.subplots()
                labels = ["Federal", "Social Security", "Medicare", "State", "Local"]
                vals = [
                    d.get("Federal Tax Withheld (Box 2)", 0.0) if d.get("Federal Tax Withheld (Box 2)") is not None else d.get("Federal Tax Withheld (Box 2)", d.get("Federal Tax Withheld (Box2)", 0.0)),
                    d.get("Social Security Tax Withheld (Box 4)", 0.0) if d.get("Social Security Tax Withheld (Box 4)") is not None else d.get("SS Withheld - Expected", 0.0),
                    d.get("Medicare Tax Withheld (Box 6)", 0.0) if d.get("Medicare Tax Withheld (Box 6)") is not None else d.get("Medicare Withheld - Expected", 0.0),
                    d.get("State Tax Withheld", 0.0),
                    d.get("Local Tax Withheld", 0.0),
                ]
                if sum(vals) <= 0:
                    ax1.text(0.5, 0.5, "No payroll withholding data to chart", ha="center", va="center")
                else:
                    ax1.pie(vals, labels=labels, autopct="%1.1f%%")
                ax1.set_title("Payroll Tax Breakdown")
                st.pyplot(fig1)

                # Income allocation bar chart
                fig2, ax2 = plt.subplots()
                labels2 = ["Gross", "Taxable After Std Ded", "Payroll Taxes", "Retirement+DCB"]
                gross = d.get("Total Gross Earnings (estimate)", 0.0)
                taxable_after = d.get("Taxable Income After Standard Deduction", 0.0)
                payroll_taxes = d.get("Total Payroll Taxes Withheld (Fed+SS+Med)", 0.0)
                retirement = d.get("Retirement Deferrals (reported)", 0.0) + d.get("Dependent Care Benefits (Box10)", 0.0)
                vals2 = [gross, taxable_after, payroll_taxes, retirement]
                ax2.bar(labels2, vals2)
                ax2.set_ylabel("Amount ($)")
                ax2.set_title("Income Allocation")
                st.pyplot(fig2)

                # Marginal bracket visualization
                fig3, ax3 = plt.subplots()
                brackets = IRS_BRACKETS.get(tax_year, IRS_BRACKETS[2024])
                xs = []
                ys = []
                for lower, upper, rate in brackets:
                    xs.append(lower)
                    ys.append(rate)
                    xs.append(upper if upper != float("inf") else (lower * 1.5 + 10000))
                    ys.append(rate)
                ax3.step(xs, ys, where='post')
                ax3.set_xlabel("Taxable Income")
                ax3.set_ylabel("Marginal Rate")
                ax3.set_title(f"Marginal Brackets ({tax_year})")
                st.pyplot(fig3)

    # Export: separate Info (original input fields) from Derived (computed fields)
    st.header("Export: Info & Derived Results")

    info_df = pd.DataFrame(info_rows)
    derived_df = pd.DataFrame(derived_rows)

    # Flatten lists for CSV in derived
    if "Detected Errors/Warnings" in derived_df.columns:
        derived_df["Detected Errors/Warnings"] = derived_df["Detected Errors/Warnings"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Info (source/input fields)")
        st.write(info_df.head(10))
        info_csv = io.StringIO()
        info_df.to_csv(info_csv, index=False)
        st.download_button("Download Info CSV", info_csv.getvalue(), file_name="w2_info.csv", mime="text/csv")

    with c2:
        st.subheader("Derived (computed fields)")
        st.write(derived_df.head(10))
        derived_csv = io.StringIO()
        derived_df.to_csv(derived_csv, index=False)
        st.download_button("Download Derived CSV", derived_csv.getvalue(), file_name="derived_w2_fields.csv", mime="text/csv")

else:
    st.info("Upload a CSV with headers matching your format (profile_id, year, employer_name, employer_id, employee_name, ssn, wages, taxable_wages, federal_income_tax_withheld, social_security_wages, social_security_tax_withheld, medicare_wages, medicare_tax_withheld, retirement_deferrals, dependent_care_benefits, state, state_wages, state_tax_withheld, local_wages, local_tax_withheld, locality_name).")

# Footer
st.markdown("---")
st.markdown("Notes: Tax brackets and deduction defaults are for 2024. Use the tax-year selector to choose other years (2021-2025). Update bracket tables if you need official 2025 values.")
