import panel as pn
import holoviews as hv
import pandas as pd
from holoviews import dim, opts
from datetime import datetime, timedelta
import calendar
from math import pi
import numpy as np

hv.extension('bokeh')
pn.extension()

from bokeh.palettes import Category20c, Category20
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.models import DatetimeTickFormatter

COLOR_MAPPING = {
    # Essentials (Yellow/Orange tones)
    'Rent': '#95A5A6',           # Cadet Grey
    'Utilities': '#778899',      # Light Slate Grey
    'Groceries': '#A9A9A9',      # Dark Gray
    'Electricity': '#696969',      # Dim Gray
    
    # Lifestyle (Red/Pink tones)
    'Dining': '#FF6B6B',        # Coral Red
    'Entertainment': '#FF1493', # Deep Pink
    'Shopping': '#DC143C',      # Crimson
    'Vacation': '#FF4444',      # Bright Red
    'Hobbies': '#FF69B4',       # Hot Pink
    
    # Financial (Green tones)
    'Investments': '#2ECC71',   # Emerald Green
    'Savings': '#3CB371',       # Medium Sea Green
    'Dividends': '#228B22',     # Forest Green
    'Interest': '#32CD32',      # Lime Green
    'Retirement': '#008000',    # Office Green
    
    # Transportation (Blue tones - added as bonus)
    'Car': '#4169E1',          # Royal Blue
    'Public Transport': '#1E90FF', # Dodger Blue
    
    # Greyish (Other categories)
    'Other': '#FFD700',          # Gold
    'unknown': '#FFA500',     # Orange
    'Transfer': '#FFE135',     # Banana Yellow
    'Cash': '#FFC000',   # Golden Yellow
    'Healthcare': '#FFD580',    # Light Orange

    "Salary": '#95A5A6',
    "Financial Support" : '#FF4444',
}

account_start_balance = 1863.44

balance_df = pd.read_csv('balances.csv', sep='\t', encoding='utf-8')  # Should have: Date, Category, Amount, Type
balance_df['Date'] = pd.to_datetime(balance_df['Date'])

# Load your transaction data (replace with your actual data source)
df = pd.read_csv('transactions.csv', sep='\t', encoding='utf-8')  # Should have: Date, Category, Amount, Type
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month_name()
df['Year'] = df['Date'].dt.year
df['Expense'] = df[df['Type'] == 'Expense']['Amount'].abs()
df['Income'] = df[df['Type'] == 'Income']['Amount']

# Create widgets
month_selector = pn.widgets.Select(name='Select Month', options=df['Month'].unique().tolist(), value='July')
year_selector = pn.widgets.Select(name='Select Year', options=df['Year'].unique().tolist(), value=2023)
category_selector = pn.widgets.Select(name='Select Category', options=df['Category'].unique().tolist())
s_date_picker = pn.widgets.DatePicker(name='Start Date', value=df['Date'].min())
e_date_picker = pn.widgets.DatePicker(name='End Date', value=df['Date'].max())
df_widget = pn.widgets.DataFrame(df, name='Transactions', width=1200)

switch = pn.widgets.Switch(name='Financial Support', value=True)
ie_switch = pn.widgets.Switch(name='Income/Expenses', value=True)

# --------------------------
# Visualization Functions
# --------------------------

def spending_by_category(month, year):
    filtered = df[df['Year'] == year]
    filtered = filtered[filtered['Month'] == month]
    data = filtered.groupby('Category')['Expense'].sum().reset_index()
    data = data[data['Expense'] != 0.00]
    data = data.sort_values(by=['Expense'])
    data['angle'] = data['Expense']/data['Expense'].sum() * 2*pi
    for category, color in COLOR_MAPPING.items():
        data.loc[data['Category'] == category, ['color']] = color

    p = figure(height=300, width=350, title="Spendings Breakdown", toolbar_location=None,
           tools="hover", tooltips="@Category: @Expense", x_range=(-0.5, 1.0))

    r = p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='Category', source=data)
    
    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    
    return p

def earnings_by_category(month, year):
    filtered = df[df['Year'] == year]
    filtered = filtered[filtered['Month'] == month]
    data = filtered.groupby('Category')['Income'].sum().reset_index()
    data = data[data['Income'] != 0.00]
    data = data.sort_values(by=['Income'])
    data['angle'] = data['Income']/data['Income'].sum() * 2*pi
    for category, color in COLOR_MAPPING.items():
        data.loc[data['Category'] == category, ['color']] = color

    p = figure(height=300, width=350, title="Earnings Breakdown", toolbar_location=None,
           tools="hover", tooltips="@Category: @Income", x_range=(-0.5, 1.0))

    r = p.wedge(x=0, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend_field='Category', source=data)
    
    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    
    return p

def breakdown_net(month, year):
    monthly_df = df.copy()
    monthly_df = monthly_df[monthly_df['Year'] == year]
    monthly_df = monthly_df[monthly_df['Month'] == month]
    expenses = monthly_df[monthly_df['Type'] == 'Expense'].sort_values(by=['Expense'])
    income = monthly_df[monthly_df['Type'] == 'Income'].sort_values(by=['Income'])
    total_expenses = expenses['Expense'].sum()
    total_income = income['Income'].sum()

    df_net = pd.DataFrame({
        'Category': ['Earnings', 'Spendings'],
        'Amount': [total_income, total_expenses]
    })
    bars = hv.Bars(df_net, 'Category', 'Amount').opts(
        title='Net Breakdown',
        color=dim('Category').categorize({'Earnings': '#2ecc71', 'Spendings': '#e74c3c'}),
        ylabel='Amount (€)',
        height=290,
        width=250,
        tools=['hover'],
        toolbar=None,
        invert_axes=False
    )
    
    return bars

def create_summary_card(title, amount, breakdown, is_income=False, is_net=False):
    color = '#2ecc71' if is_income else '#e74c3c' if not is_net else '#2ecc71' if amount>0 else '#e74c3c'
    icon = '💵' if is_income else '💰' if not is_net else '⚖️'
    
    if is_income:
        graph = pn.bind(earnings_by_category, month=month_selector, year=year_selector)
    elif is_net:
        graph = pn.bind(breakdown_net, month=month_selector, year=year_selector)
    else:
        graph = pn.bind(spending_by_category, month=month_selector, year=year_selector)
    
    return pn.Card(
        pn.Column(
            pn.Row(pn.pane.Str(icon, styles={'font-size': '1.5em'}), pn.pane.Markdown(f"**{title}**")),
            pn.pane.HTML(f"<div style='font-size: 2em; font-weight: bold; color: {color}; margin: 10px 0;'>€{amount:,.2f}</div>"),
            graph,
            styles={'padding': '10px'}
        ),
        styles={'box-shadow': '0 2px 4px rgba(0,0,0,0.05)', 'margin': '10px'}
    )

def get_monthly_data(month, year):
    monthly_df = df.copy()
    monthly_df = monthly_df[monthly_df['Year'] == year]
    monthly_df = monthly_df[monthly_df['Month'] == month]
    expenses = monthly_df[monthly_df['Type'] == 'Expense'].sort_values(by=['Expense'])
    income = monthly_df[monthly_df['Type'] == 'Income'].sort_values(by=['Income'])
    # Create breakdown dictionaries
    expense_breakdown = expenses.groupby('Category')['Expense'].sum().to_dict()
    income_breakdown = income.groupby('Category')['Income'].sum().to_dict()
    # Calculate totals
    total_expenses = expenses['Expense'].sum()
    total_income = income['Income'].sum()

    return {
        'expenses': {
            'total': total_expenses,
            'breakdown': expense_breakdown
        },
        'income': {
            'total': total_income,
            'breakdown': income_breakdown
        }
    }


def monthly_summary(month, year):
    data = get_monthly_data(month, year)
    net = data['income']['total'] - data['expenses']['total']
    
    expense_card = create_summary_card(
        "Total Expenses", 
        data['expenses']['total'], 
        data['expenses']['breakdown']
    )
    
    income_card = create_summary_card(
        "Total Income",
        data['income']['total'],
        data['income']['breakdown'],
        is_income=True
    )
    
    net_card = create_summary_card(
        "Net Balance",
        net,
        {},
        is_net=True
    )
    
    # Create responsive grid
    return pn.FlexBox(
        expense_card, 
        income_card, 
        net_card,
        flex_wrap='wrap',
        justify_content='center',
        styles={'gap': '20px', 'padding': '20px'}
    )

def monthly_trends():
    monthly = df.groupby(['Year', 'Month'])['Expense'].sum().reset_index()
    monthly['Date'] = monthly.apply(lambda x: datetime(x['Year'], 
                                   list(calendar.month_name).index(x['Month']), 1), 1)
    return hv.Curve(monthly, 'Date', 'Expense').opts(
        title='Monthly Spending Trends', width=800, height=300)

def income_vs_expenses(start, end, support):
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)
    ive_df = df.copy()
    ive_df = ive_df[ive_df['Date'] >= pd_start]
    ive_df = ive_df[ive_df['Date'] <= pd_end]

    if not support:
        ive_df = ive_df[ive_df['Category'] != 'Financial Support']

    ive_df = ive_df.groupby(['Year', 'Month']).agg({'Income':'sum', 'Expense':'sum'}).reset_index()
    ive_df['Year'] = ive_df['Year'].astype(str)
    ive_df["Period"] = ive_df["Month"] + ' ' + ive_df["Year"]
    ive_df["Period"] = pd.to_datetime(ive_df["Period"], format="%B %Y")
    ive_df = ive_df.sort_values(by=['Period'])
    bars = hv.Curve(ive_df, 'Period', 'Expense').opts(color='red', tools=['hover'])
    line = hv.Curve(ive_df, 'Period', 'Income').opts(color='green', tools=['hover'])

    return (bars * line).opts(
        title='Income vs Expenses', width=500, height=350)

def account_balance_evolution(start, end, support):
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)
    b_df = balance_df.copy()
    b_df = b_df[b_df['Date'] >= pd_start]
    b_df = b_df[b_df['Date'] <= pd_end]

    if support:
        line = hv.Curve(b_df, 'Date', 'Balance').opts(
            color='black',
            tools=['hover'],
            )
    else:
        line = hv.Curve(b_df, 'Date', 'WithoutSupport').opts(
            color='black',
            tools=['hover'],
            )

    return (line).opts(
        title='Account Balance', width=500, height=350)

def create_overview_card(title, cat1, cat2, cat3, amount, amount2, amount3, is_balance=False):
    color = '#2ecc71' if amount > 0 else '#e74c3c'
    icon = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="2" x2="12" y2="22"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H7"></path></svg>' if is_balance else '⚖️'
    
    if is_balance:
        graph = pn.bind(account_balance_evolution, start=s_date_picker, end=e_date_picker, support=switch)
    else:
        graph = pn.bind(income_vs_expenses, start=s_date_picker, end=e_date_picker, support=switch)
    
    return pn.Card(
    pn.Column(
        # Header with icon and title
        pn.Row(
            pn.pane.Str(icon, styles={'font-size': '1.5em', 'margin-right': '10px'}),
            pn.pane.Markdown(f"**{title}**", styles={'font-size': '1.2em', 'color': '#333'}),
            align='center'
        ),
        
        # Values display
        pn.Row(
            pn.Column(
                pn.pane.Markdown(cat1, styles={'font-size': '1em', 'color': '#666'}),
                pn.pane.HTML(f"<div style='font-size: 1.5em; font-weight: bold; color: {color};'>€{amount:,.2f}</div>"),
                align='center'
            ),
            pn.Column(
                pn.pane.Markdown(cat2, styles={'font-size': '1em', 'color': '#666'}),
                pn.pane.HTML(f"<div style='font-size: 1.5em; font-weight: bold; color: #e74c3c;'>€{amount2:,.2f}</div>"),
                align='center'
            ),
            pn.Column(
                pn.pane.Markdown(cat3, styles={'font-size': '1em', 'color': '#666'}),
                pn.pane.HTML(f"<div style='font-size: 1.5em; font-weight: bold; color: #2ecc71;'>€{amount3:,.2f}</div>"),
                align='center'
            ),
            #justify_content='space-between',
            #margin='10px 0'
        ),
        
        # Graph
        graph,
        
        styles={'padding': '20px', 'border-radius': '8px'}
    ),
    styles={
        'box-shadow': '0 4px 8px rgba(0,0,0,0.1)',
        'margin': '10px',
        'border': '1px solid #e0e0e0',
        'background': '#ffffff'
    },
)

def overview_summary(start, end, support):
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)

    b_df = balance_df.copy()
    b_df = b_df[b_df['Date'] >= pd_start]
    b_df = b_df[b_df['Date'] <= pd_end]

    ive_df = df.copy()
    ive_df = ive_df[ive_df['Date'] >= pd_start]
    ive_df = ive_df[ive_df['Date'] <= pd_end]

    if support:
        balance = b_df['Balance'].tail(1).item()
        min_balance=b_df['Balance'].min()
        max_balance=b_df['Balance'].max()

        total_expenses = ive_df['Expense'].sum()
        total_income = ive_df['Income'].sum()
        net = total_income - total_expenses
    else:
        balance = b_df['WithoutSupport'].tail(1).item()
        min_balance=b_df['WithoutSupport'].min()
        max_balance=b_df['WithoutSupport'].max()

        ive_df = ive_df[ive_df['Category'] != 'Financial Support']
        total_expenses = ive_df['Expense'].sum()
        total_income = ive_df['Income'].sum()
        net = total_income - total_expenses

    incomve_vs_expense_card = create_overview_card(
        "Transactions",
        "Net Balance",
        "Total Expenses",
        "Total Income", 
        net, 
        total_expenses,
        total_income,
    )
    
    balance_card = create_overview_card(
        "Bank Account Balance",
        "Current",
        "Min",
        "Max",
        balance,
        min_balance,
        max_balance,
        is_balance=True
    )
    
    # Create responsive grid
    return pn.FlexBox(
        incomve_vs_expense_card, 
        balance_card, 
        flex_wrap='wrap',
        justify_content='center',
        styles={'gap': '0px', 'padding': '20px'}
    )

def category_comparison(start, end, category, mode):
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)
    cat_df = df.copy()
    cat_df = cat_df[cat_df['Date'] >= pd_start]
    cat_df = cat_df[cat_df['Date'] <= pd_end]
    cat_df = cat_df[cat_df['Category'] == category]
    cat_df = cat_df.groupby(['Year', 'Month']).agg({'Income':'sum', 'Expense':'sum'}).reset_index()
    cat_df['Year'] = cat_df['Year'].astype(str)
    cat_df["Period"] = cat_df["Month"] + ' ' + cat_df["Year"]
    cat_df["Period"] = pd.to_datetime(cat_df["Period"], format="%B %Y")
    cat_df = cat_df.sort_values(by=['Period'])
    print(cat_df.tail(20))
    if mode:
        bars = hv.Bars(cat_df, 'Period', 'Income').opts(color='green', tools=['hover'], width=1200, height=400)
    else:
        bars = hv.Bars(cat_df, 'Period', 'Expense').opts(color='red', tools=['hover'], width=1200, height=400)

    return bars

def create_category_card(title, amount, is_income):
    color = '#2ecc71' if is_income else '#e74c3c'
    icon = '💵' if is_income else '💰'
    
    graph = pn.bind(category_comparison, start=s_date_picker, end=e_date_picker, category=category_selector, mode=ie_switch)

    return pn.Card(
        pn.Column(
            pn.Row(pn.pane.Str(icon, styles={'font-size': '1.5em'}), pn.pane.Markdown(f"**{title}**")),
            pn.pane.HTML(f"<div style='font-size: 2em; font-weight: bold; color: {color}; margin: 10px 0;'>€{amount:,.2f}</div>"),
            graph,
            styles={'padding': '10px'}
        ),
        styles={'box-shadow': '0 2px 4px rgba(0,0,0,0.05)', 'margin': '10px'}
    )

def category_breakdown(start, end, category, mode):
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)
    cat_df = df.copy()
    cat_df = cat_df[cat_df['Date'] >= pd_start]
    cat_df = cat_df[cat_df['Date'] <= pd_end]
    cat_df = cat_df[cat_df['Category'] == category]
    cat_df = cat_df.groupby(['Year', 'Month']).agg({'Income':'sum', 'Expense':'sum'}).reset_index()

    if mode:
        amount = cat_df['Income'].sum()
        title = "Incomes generated by " + category
        is_income = True
    else:
        amount = cat_df['Expense'].sum()
        title = "Expenses generated by " + category
        is_income = False

    category_card = create_category_card(
        title,
        amount,
        is_income
    )
    
    # Create responsive grid
    return pn.FlexBox(
        category_card, 
        flex_wrap='wrap',
        justify_content='center',
        styles={'gap': '0px', 'padding': '20px'}
    )

def cumulative_spending():
    cumulative = df.sort_values('Date').assign(
        Cumulative=df['Expense'].cumsum())
    return hv.Area(cumulative, 'Date', 'Cumulative').opts(
        title='Cumulative Spending', width=800, height=300)

def budget_vs_actual():
    # Requires a budget DataFrame with 'Category' and 'Budget' columns
    actual = df.groupby('Category')['Expense'].sum().reset_index()
    merged = pd.merge(actual, budget_df, on='Category')
    return hv.Bars(merged, ['Category', 'Budget', 'Expense']).opts(
        title='Budget vs Actual', width=800, height=400)

def top_expenses(month):
    filtered = df[(df['Month'] == month) & (df['Expense'] > 0)]
    return hv.Table(filtered.nlargest(10, 'Expense')).opts(
        title='Top 10 Expenses', width=600)

def spending_heatmap():
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    heatmap_data = df.pivot_table(index='DayOfWeek', 
                                columns='Week', 
                                values='Expense', 
                                aggfunc='sum')
    return hv.HeatMap(heatmap_data).opts(
        title='Weekly Spending Pattern', width=800, height=300)

def net_worth_timeline():
    # Requires assets/liabilities data
    net_worth = df.groupby('Date')['Amount'].sum().cumsum().reset_index()
    return hv.Curve(net_worth, 'Date', 'Amount').opts(
        title='Net Worth Over Time', width=800, height=300)

@pn.depends(ie_switch.param.value)
def status_indicator(value):
    return pn.pane.Markdown(f"**Mode:** {'Income' if value else 'Expense'}")

# --------------------------
# Dashboard Layout
# --------------------------

dashboard = pn.Tabs(
    ('Overview', pn.Column(
        pn.pane.Markdown("# Financial Overview", styles={'text-align': 'center'}),
        pn.Row(s_date_picker, e_date_picker, pn.Column(pn.pane.HTML('<label>Financial Support:</label>'), switch)),
        pn.Row(pn.bind(overview_summary, start=s_date_picker, end=e_date_picker, support=switch)),

    )),
    ('Monthly', pn.Column(
        pn.pane.Markdown("# Monthly Financial Summary", styles={'text-align': 'center'}),
        pn.Row(month_selector, year_selector),
        pn.bind(monthly_summary, month=month_selector, year=year_selector),
        styles={'background': '#f9f9f9', 'border-radius': '12px', 'padding': '20px'}
    )),
    ('Categories', pn.Column(
        pn.pane.Markdown("# Category Breakdown", styles={'text-align': 'center'}),
        pn.Row(s_date_picker, e_date_picker, category_selector, pn.Column(status_indicator, ie_switch)),
        pn.bind(category_breakdown, start=s_date_picker, end=e_date_picker, category=category_selector, mode=ie_switch),
    )),
    ('Data', pn.Column(
        pn.pane.Markdown("# Transactions Dataframe", styles={'text-align': 'center'}),
        df_widget
    )),
)

# Add interactivity
dashboard = dashboard.servable()
#dashboard.servable().show()
