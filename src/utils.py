import pandas as pd
import numpy as np


ORIG_COLS = [
    'credit_score', 'first_payment_date', 'first_time_homebuyer',
    'maturity_date', 'msa', 'mip', 'units', 'occupancy_status',
    'ocltv', 'dti', 'orig_upb', 'oltv', 'orig_interest_rate',
    'channel', 'ppm_flag', 'amortization_type', 'property_state',
    'property_type', 'zip_code', 'loan_seq_num', 'loan_purpose',
    'orig_loan_term', 'num_borrowers', 'seller_name', 'servicer_name',
    'super_conforming_flag', 'pre_relief_refinance_seq_num',
    'program_indicator', 'relief_refinance_indicator',
    'property_valuation_method', 'io_indicator', 'mi_cancellation'
]

SVCG_COLS = [
    'loan_seq_num', 'monthly_reporting_period', 'current_upb',
    'current_loan_delinquency_status', 'loan_age', 'remaining_months_to_maturity',
    'repurchase_flag', 'modification_flag', 'zero_balance_code',
    'zero_balance_effective_date', 'current_interest_rate',
    'current_deferred_upb', 'due_date_last_paid_installment',
    'mi_recoveries', 'net_sales_proceeds', 'non_mi_recoveries',
    'expenses', 'legal_costs', 'maintenance_preservation_costs',
    'taxes_insurance', 'misc_expenses', 'actual_loss',
    'modification_cost', 'step_modification_flag',
    'deferred_payment_plan', 'estimated_ltv', 'zero_balance_removal_upb',
    'delinquent_accrued_interest', 'delinquency_due_disaster',
    'borrower_assistance_status_code', 'current_month_modification_cost',
    'interest_bearing_upb'
]

SENTINEL_VALUES = {
    'credit_score': 9999,
    'dti': 999,
    'ocltv': 999,
    'oltv': 999
}

DROP_COLS = [
    'pre_relief_refinance_seq_num',
    'relief_refinance_indicator',
    'super_conforming_flag'
]


def load_origination(data_dir: str, years: list) -> pd.DataFrame:
    """Load and combine origination files for given years."""
    frames = []
    for year in years:
        df = pd.read_csv(
            f'{data_dir}/sample_orig_{year}.txt',
            sep='|', header=None, names=ORIG_COLS, low_memory=False
        )
        df['year'] = year
        frames.append(df)
        print(f"  Loaded origination {year}: {len(df):,} loans")
    return pd.concat(frames, ignore_index=True)


def load_performance(data_dir: str, years: list) -> pd.DataFrame:
    """Load and combine performance files for given years."""
    frames = []
    for year in years:
        df = pd.read_csv(
            f'{data_dir}/sample_svcg_{year}.txt',
            sep='|', header=None, names=SVCG_COLS, low_memory=False
        )
        frames.append(df)
        print(f"  Loaded performance {year}: {len(df):,} records")
    svcg = pd.concat(frames, ignore_index=True)
    svcg['current_loan_delinquency_status'] = pd.to_numeric(
        svcg['current_loan_delinquency_status'], errors='coerce'
    )
    return svcg


def clean_origination(orig):
    for col, sentinel in SENTINEL_VALUES.items():
        if col in orig.columns:
            orig[col] = orig[col].replace(sentinel, np.nan)
    cols_to_drop = [c for c in DROP_COLS if c in orig.columns]
    if cols_to_drop:
        orig = orig.drop(columns=cols_to_drop)
    return orig


def create_target(orig: pd.DataFrame, svcg: pd.DataFrame,
                  dpd_threshold: int = 3) -> pd.DataFrame:
    """
    Create binary default target variable.
    Default = loan ever reached dpd_threshold+ months delinquent.
    """
    default_loans = svcg[
        svcg['current_loan_delinquency_status'] >= dpd_threshold
    ]['loan_seq_num'].unique()

    orig['default'] = orig['loan_seq_num'].isin(default_loans).astype(int)
    print(f"Default rate: {orig['default'].mean()*100:.2f}%")
    return orig


def load_data(data_dir: str, years: list) -> pd.DataFrame:
    """
    Master function — load, clean, and return modeling-ready dataframe.
    Usage: from src.utils import load_data
           df = load_data('../data', [2022, 2023])
    """
    print("Loading origination data...")
    orig = load_origination(data_dir, years)

    print("\nCleaning origination data...")
    orig = clean_origination(orig)

    print("\nLoading performance data...")
    svcg = load_performance(data_dir, years)

    print("\nCreating target variable...")
    orig = create_target(orig, svcg)

    print(f"\nDone! Dataset: {orig.shape}")
    return orig


def load_performance_data(data_dir: str, years: list) -> pd.DataFrame:
    """
    Load performance (servicing) data.
    Usage: from src.utils import load_performance_data
           svcg = load_performance_data('../data', [2022, 2023])
    """
    print("Loading performance data...")
    svcg = load_performance(data_dir, years)
    svcg['monthly_reporting_period'] = pd.to_datetime(
        svcg['monthly_reporting_period'], format='%Y%m'
    )
    print(f"Done! Performance dataset: {svcg.shape}")
    return svcg