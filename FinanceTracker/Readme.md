# Financial Tracker Documentation

## Overview
The parsing script analyses bank records (stored as pdfs in the pdf-folder). It extracts all transactions from the bank records and groups them into spending categories.
The dashboard provides interactive visualization and analysis of personal financial data, including income, expenses, and account balance tracking. Built with Panel and Holoviews, it features:

- Multi-tab interface for different analysis perspectives
- Interactive filters for date ranges and categories
- Financial metric cards with embedded visualizations
- Real-time data updates

## Data Requirements
### Required CSV Files:
1. `transactions.csv` - Financial transactions
   Columns:
   - Date (YYYY-MM-DD)
   - Category (string)
   - Amount (float)
   - Type (Income/Expense)

2. `balances.csv` - Account balance history
   Columns:
   - Date (YYYY-MM-DD)
   - Balance (float)
   - WithoutSupport (float)

### Data Preparation:
- Sample transaction data format
  ```python
  transactions = pd.DataFrame({
      'Date': ['2023-01-01', '2023-01-05'],
      'Category': ['Groceries', 'Salary'],
      'Amount': [-150.00, 3000.00],
      'Type': ['Expense', 'Income']
  })

## Main Components
1. Color Mapping
  - **Essentials**: Gray/Yellow tones
  - **Lifestyle**: Red/Pink tones
  - **Financial**: Green tones
  - **Transport**: Blue tones

2. Core Visualization Functions
| Function      | Description | Output  |
| ----------- | ----------- | ----------- |
| `spending_by_category()` | Pie chart of expense distribution | Bokeh Figure |
| `earnings_by_category()`   | Pie chart of income sources | Bokeh Figure |
| `income_vs_expenses()`   | Dual-axis income/expense trend	 | Holoviews Overlay |
| `account_balance_evolution()`   | Balance timeline chart | Holoviews Curve |

3. Interactive Widgets
- Breakdown of the Widgets used in the Dashboard
  ```python
  # Date controls
  s_date_picker = pn.widgets.DatePicker(name='Start Date')
  e_date_picker = pn.widgets.DatePicker(name='End Date')

  # Category selection
  category_selector = pn.widgets.Select(name='Select Category')
  
  # Financial toggles
  switch = pn.widgets.Switch(name='Financial Support')

4. Dashboard Layout
- Breakdown of the Tabs available in the Dashboard
  ```python
  dashboard = pn.Tabs(
      ('Overview', ...),
      ('Monthly', ...), 
      ('Categories', ...),
      ('Data', ...)
  )

## Key Features
1. **Dynamic Filtering**
  - Date range selection
  - Month/year pickers
  - Category-specific analysis

2. **Financial Cards**
  - Value displays with color-coding
  - Embedded mini-charts
  - Responsive layout

3. **Visualization Types**
  - Pie charts (category breakdown)
  - Line charts (trend analysis)
  - Bar charts (comparisons)
  - Heatmaps (spending patterns)

## Usage
1. **Install Dependencies**
  - `pip install panel holoviews pandas bokeh`

2. **Run Dashboard**
  - `panel serve finance_dashboard.py --show`

3. **Interact with Components**
  - Dashboard Interface
