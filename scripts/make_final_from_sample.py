import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
csv = root / 'sample.csv'
TAX_RATES = {"ss_rate": 0.062, "medicare_rate": 0.0145}

df = pd.read_csv(csv)
final = pd.DataFrame()
final['year'] = 2025
final['employer_name'] = ''
final['employer_id'] = ''
final['employee_name'] = df.get('Employee Name', df.columns[0])
final['ssn'] = ''
final['wages'] = pd.to_numeric(df.get('Wages'), errors='coerce')
final['taxable_wages'] = final['wages']
final['federal_income_tax_withheld'] = pd.to_numeric(df.get('FederalTaxWithheld'), errors='coerce')
final['social_security_wages'] = pd.to_numeric(df.get('SocialSecurityWages'), errors='coerce')
final['social_security_tax_withheld'] = pd.to_numeric(df.get('SocialSecurityTax'), errors='coerce')
final['medicare_wages'] = pd.to_numeric(df.get('MedicareWages'), errors='coerce')
final['medicare_tax_withheld'] = pd.to_numeric(df.get('MedicareTax'), errors='coerce')
final['retirement_deferrals'] = pd.to_numeric(df.get('401kContrib'), errors='coerce')
final['dependent_care_benefits'] = pd.to_numeric(df.get('DependentCareBenefits'), errors='coerce')
final['state'] = ''
final['state_wages'] = pd.to_numeric(df.get('StateWages'), errors='coerce')
final['state_tax_withheld'] = pd.to_numeric(df.get('StateTaxWithheld'), errors='coerce')
final['local_wages'] = ''
final['local_tax_withheld'] = ''
final['locality_name'] = ''
final['net_pay'] = final['wages'] - final['federal_income_tax_withheld'].fillna(0) - final['social_security_tax_withheld'].fillna(0) - final['medicare_tax_withheld'].fillna(0)
final['social_security_tax_rate'] = ('{:.4f}'.format(TAX_RATES['ss_rate']).rstrip('0').rstrip('.'))
final['medicare_tax_rate'] = ('{:.4f}'.format(TAX_RATES['medicare_rate']).rstrip('0').rstrip('.'))

out = root / 'parsed_w2_standard_sample.csv'
# Round numeric columns sensibly before writing sample CSV: monetary to 2 decimals, tax rates to 4 decimals
try:
	export_df = final.copy()
	num_cols = export_df.select_dtypes(include=['number']).columns.tolist()
	if num_cols:
		export_df.loc[:, num_cols] = export_df.loc[:, num_cols].round(2)
	rate_cols = ['social_security_tax_rate', 'medicare_tax_rate']
	for rc in rate_cols:
		if rc in export_df.columns:
			try:
				export_df[rc] = export_df[rc].apply(lambda v: ("{:.4f}".format(v)).rstrip('0').rstrip('.') if pd.notnull(v) else "")
			except Exception:
				export_df[rc] = export_df[rc].round(4)
	export_df.to_csv(out, index=False)
except Exception:
	# fallback to default write
	final.to_csv(out, index=False, float_format='%.2f')
print('Wrote', out)
print(final.head().to_string(index=False))
