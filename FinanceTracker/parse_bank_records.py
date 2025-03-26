import pandas as pd
import pdfplumber
import os
import re
from datetime import datetime, timedelta
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

# --------------------------
# Constants and Configuration
# --------------------------

PRIMARY_CATEGORIES = {
    "Essentials": ["Rent", "Utilities", "Groceries"],
    "Lifestyle": ["Dining", "Entertainment", "Shopping"],
    "Financial": ["Investments", "Savings", "Debt Payments"]
}

CATEGORY_MAPPING = {
    "Rent": ['Sibylle Schwarzmann', 'HABYT', 'HOUSINGANY', 'PROJECTS CO-LIVING'],
    "Salary": ['Lohn/Gehalt'],
    "Financial Support" : ['Harald Stadelmann', 'Claudia Stadelmann'],
    "Investments" : ['Bausparkasse Schwaebisch Hall', 'Trade Republic', 'Dividende', 'Gutschrift', 'Echtzeitüberweisung', 'Sollzins', 'Jakob Stadelmann'],
    "Groceries" : ['EDEKA', 'NETTO', 'BAECKER', 'Backerei', 'BACKHAUS', 'BACK', 'KAUFLAND', 'Beck', 'BIOMARKT', 'LOKALBAeCKEREI', 'BROTZEIT', 'VILLAGE', 'LOsteria', 'NORMA', 'E CENTER', 'SPAR', 'BILLA', 'NAH + GUT', 'EXPALAMEDAII','TRES CRUCES', 'EXPRESS MONTERA', 'PRIMAPRIX', 'EXPRESS', 'LEVADURA', 'EXPLEON', 'DIA', 'EXPRESS', 'ALBERT HEIJN', 'Jodenbree', 'FROMAGERIE', 'LIDL', 'ANDRES FELIPE PEREZ', 'SUPERMERCADOS', 'EXPSANTAISABEL','MARKET', 'SODEXO', 'MERCADONA', 'WU-CHEN', 'EXPLEON', 'PAZO DE LUGO', 'SUPER DIAMAK', 'ALIMENTACION', 'PANETTERIA', 'EL CORTE INGLES', 'JOKANILO', 'BAZAR', 'MULAS', 'YAKARTA', 'EXPGRAVINA3', 'FURONG CHEN','EXPPRINCIPEIII', 'PIGNIC', 'BOULANG', 'OUARIBI', 'FRANPRIX', 'AUX', 'MONOPRIX', 'MARTIN', 'G20 GRENIER', 'EXPSANMIGUEL', 'FYV MANOLI', 'FRUTERIA', 'BAKEHOUSE', 'EXPNTOLOSA', 'ALEGAS', 'YORMAS', 'STUDENTENWERK', 'Backmuehle', 'REWE','ZEIT FUER BROT', 'ALDI', 'GRUENHOFFS BACKSTUUV', 'BackWerk', 'SUPERMERCADO', 'ALIMENTACIO', 'Hofer', 'CITY MARKT', 'BOUL', 'EXPRUAMAYOR2', 'DALE STAER GROUPSL'],
    "Dining" : ['Burger','Reimanns', 'Kitzmann', 'PIZZETTA', 'PIZZA', 'PIZZERIA', 'Arizona', 'Gastronomie', 'KEBAB', 'KEBAP', 'BURGERBAR', 'Schnitzel', 'RESTAURANT', 'FELIZ', 'POKE AND GREENS', 'PAPIZZA', 'MISSJIANBING', 'Kitchen', 'Lente', 'Lagom',  'CAFE', 'PLENTI', 'FALAFELERIA', 'PICCOLO', 'PICCOLA', 'TABERNA', 'CHAPANDAZ', 'ZONA ROSA', 'RAMEN', 'PUPUSERIA', 'COFFEE', 'COFFE', 'SALT IN CAKE', 'CHOCOLAT', 'ENTRE SANTOS', 'FRUTIDIMARE', 'Las Fritas', 'DEMASIE', 'PERSA', 'RESTAURANTE','IL REGNO DI NAPOLI', 'MONTERA CLEO', 'COCOHOUSE', 'GOAT', 'GROSSO NAPOLETANO', 'CARRE PAIN', 'MANTEIGARIA', 'CAFES', 'BURGUER', 'HORCHATERIA', 'PROSCIUTTERIA', 'CASA ANGEL', 'CELICIOSO', 'THE COOKIE LAB', 'BRESCA', 'MUNE', 'TRATTORIA','Caf FRED', 'TABERNA', 'Bao House', 'FUNDACAO', 'MCDONALDS', 'Le Xuan Dac', 'Ba Tho', 'Burgermeister', 'OSTERIA', 'MARCOS', 'D?ANTONI', 'MICHAELIGARTEN', 'Hokey Pokey', 'MI ALCAMPO RIBERA DE', 'CCV*BIS', 'CAPI CORP EU'],
    "Car" : ['TANKEN', 'Tank', 'PARKHSVERWALTUN', 'PARK', 'Tiefgarage', 'JET', 'Erik Walther', 'Worldline Sweden AB fuer Shell', 'PARKEN', 'kassetten radio bmw', 'STW ER-NBG. AUFW.', 'ARAL', 'Tankstelle', 'AUTODOC', 'Wildschaden', 'REIFEN BRANDT', 'OMV', 'LM Energy', 'ROEDL'],
    "Vacation" : ['Airbnb', 'AREAS SANTS ESTACION', 'PARK GELL', 'SANTS ESTACIO', 'BALMES', 'EASYJET', 'LIVINGLISBOA', 'AEROPORTO/LISBOA', 'ESKY', 'Generator Berlin', 'Henri Sturm', 'Johannes Stroebel'],
    "Entertainment" : ['NBA Digital Services Intern', 'Ticketmaster', 'Krasser Stoff Merchandising', 'Albertina', 'LIBRERIA', 'museumple', 'SODEXO', 'ENTRADAS SPORT', 'IMPULSA EVENTOS', 'YELMO FILMS', 'SMART INSIDERS', 'BOOKS', 'BARRO', 'LIBROS', 'MUSEUM', 'Tickets', 'ticket', 'KUNST-WERKE', 'Audi Dome', 'HUGENDUBEL', 'Ikarus Festival', 'SARAH SILLIS', 'WEEZEVENT', 'ANTIQUARIAT', 'City Bowling UG Ansbach', 'CURIOSITYSTREAM', 'NEBULA', 'NBA', 'E-WERK', 'green and bean', 'BAR CELONA', 'Celona Nuernberg', 'Bulldog', 'TOY ROOM', 'QW+', 'LOOP', 'CERVECERIA', 'MACANUDO BAR', 'SHAPLA', 'AMOR VOODOO', 'GABRIEL', 'DICE', 'KARAOKE', 'MOVIDACLUBPAR', 'MUCHO BAR', 'SUPERSONIC', 'LA CHATA', 'DRINKS', 'Getranke Hoffmann', 'HOTEL IM EUROPA-CENTER', 'CLUB', 'ELEMENT TAXI', 'BAHNWAeRTER THIEL', 'NIGHTOWL', 'ENCHILADA', 'Acapulco', 'Carderobe', 'An einem Sonntag im Au/Berlin', 'Mono-Loco', 'MW Freimann Betriebs GmbH', 'LISBOA BAR AM BOXI'],
    "Utilities" : ['Stadtwerke Ingolstadt', 'Apotheke', 'MUELLER', 'Spitzenwerk', 'DB Vertrieb', 'Praxis', 'metro', 'RENFE', 'OFICINA', 'YOIGO', 'HAIR', 'PRIMOR', 'FCIA.', 'EREF: YYW1030390', 'TREATWELL', 'FARMACIA', 'EREF: YYW1030390', 'SNCF', 'FGV', 'BKG*BOOKING.COM FLIGHT', 'IBERIA0004402088987', 'CALLE MAYOR', 'BAUHAUS', 'DM', 'MANOMANO','TEDI', 'SATURN', 'INTENDENTE', 'CAIS SODRE', 'GEBERS', 'POLIZEI-SPORT-VEREIN', 'DECATHLON', 'New Balance Athletic Shoes', 'SPORT', 'HM 626 SAGT VIELEN DANK', 'Staatsoberkasse Bayern', 'Landesamt fur Finanzen Dienststelle'],
    "Cash" : ['Auszahlung', 'GRUPO CAJAMAR', 'LCL'],
    "Shopping" : ['PayPal', 'Episode', 'Unvain Studios', 'Zeitgeist', 'Retroschatz', 'TEXTILE', 'HUMANA', 'EL TEMPLO DE SUSU', 'TEXT DESIGN', 'WILLIAMSBURG', 'VINTAGE', 'LOLINA', 'MAGPIE', 'EXOTICA', 'Deloox', 'BSTN', 'HIPPY MARKET', 'ZIBA', 'D.B. SHOP', 'KS1', 'RAQUEL VENTURA', 'THE GOOD LUCK FACTORY', 'Sole Brothers', 'Lumpenbande', 'Kamera', 'ebay', 'Ihr Einkauf bei Ben ER', 'ReSales', 'Fabric Sales', 'Clothing', 'Deluxeboxen', 'Notino', 'Darpdecade', 'VinoKilo', 'TX SPORTS', 'ebay', 'Brosche Galvanowerk', 'Eastside Brillen', 'SUPERCONSCIOUS', 'PICKNWEIGHT', 'BIVIO', 'SNIPES', 'Amazon', 'Vinted', 'schachbrett', 'Sofie Schwegele']
}

PRE_COMPILED_REGEX = {
    'transaction_value': re.compile(r'(-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2})\s([HS])'),
    'transaction_date': re.compile(r'([0-3][0-9].[0-1][0-9].20[0-9][0-9])'), 
    'clean_value': re.compile(r'[^\d-]')
}

# --------------------------
# PDF Processing
# --------------------------
def read_pdf(path: str) -> str:
    """Optimized PDF text extraction with page range handling"""
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(keep_blank_chars=True)
                if page_text:
                    text_parts.append(process_page_text(page_text, page.page_number, len(pdf.pages)))
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
    return ''.join(text_parts)

def process_page_text(text: str, page_num: int, total_pages: int) -> str:
    """Handle page-specific text processing"""
    if page_num == 1:
        return process_first_page(text)
    elif page_num > 1 and page_num < total_pages - 1:
        return process_middle_page(text)
    elif page_num == total_pages - 1:
        return process_last_page(text)
    else:
        return ""
    
def process_first_page(text: str) -> str:
    posBegin = re.search('alter Kontostand', text)
    text = text[posBegin.start():]
    try:
        posEnd = re.search('Übertrag auf Blatt', text)
        text = text[:posEnd.start()]
    except:
        posEnd = re.search('neuer Kontostand', text)
        text = text[:posEnd.start()]
    return text

def process_last_page(text: str) -> str:
    posBegin = re.search('Übertrag von Blatt', text)
    text = text[posBegin.start():]
    posEnd = re.search('neuer Kontostand', text)
    text = text[:posEnd.start()]
    return text

def process_middle_page(text: str) -> str:
    posBegin = re.search("Übertrag von Blatt", text)
    text = text[posBegin.start():]
    posEnd = re.search("Übertrag auf Blatt", text)
    text = text[:posEnd.start()]
    return text

# --------------------------
# Transaction Processing
# --------------------------
def categorize_transaction(text: str) -> str:
    """Optimized category lookup with pre-compiled patterns"""
    text_upper = text.upper()
    for category, keywords in CATEGORY_MAPPING.items():
        if any(kw.upper() in text_upper for kw in keywords):
            return category
    return "unknown"

@lru_cache(maxsize=1000)
def parse_date(date_str: str) -> Tuple[str, str, str]:
    """Cached date parsing with datetime"""
    try:
        dt = datetime.strptime(date_str, "%d.%m.%Y")
        return (
            str(dt.day),
            dt.strftime("%B"),
            str(dt.year)
        )
    except ValueError:
        return ("unknown", "unknown", "unknown")

# --------------------------
# Data Extraction
# --------------------------
def check_semester_abroad(text, date):
    semester_abroad = ['HABYT', 'MADRID', 'HOUSINGANY', 'GETAFE', 'Staatsoberkasse Bayern']
    date_strt = datetime(2023, 8, 23)
    date_end = datetime(2024, 1, 29)
    dt = datetime.strptime(date, '%d.%m.%Y')

    for s in semester_abroad:
        if s.upper() in text.upper() and dt >= date_strt and dt <= date_end:
            return True
    return False

def check_vacation(text):
    vacation = ['Amsterdam', 'AMSTERDAM', 'WIEN', 'BARCELONA', 'PARIS', 'LISBOA', 'LIVINGLISBOA', 'BERLIN', 'MILANO', 'NEW YORK', 'CHICAGO']

    for va in vacation:
        if va.upper() in text.upper():
            return True
    return False

def process_transaction(transaction: Dict, year: str) -> Dict:
    """Optimized transaction processing with vectorized operations"""
    full_text = transaction['header'] + ''.join(transaction['description'])
    
    # Value extraction
    value = re.search(r'((\d\d\d.\d\d\d,|\d\d.\d\d\d,|\d.\d\d\d,|\d\d\d,|\d\d,|\d,)\d\d (H|S))', full_text)
    value = value.group()
    value = value.replace(".", "")
    value = value.replace(",", ".")
    value_numeric = float(value[:-2])
    if value[-1] == 'S':
        spendingtype = 'Expense'
        value_numeric = value_numeric * -1
    else: 
        spendingtype = 'Income'
    
    # Date processing
    date_match = PRE_COMPILED_REGEX['transaction_date'].search(full_text)
    if date_match:
        date_info = parse_date(date_match.group(0))
        date = date_match.group(0)
    
    if not date_match or date_info == ("unknown", "unknown", "unknown"):
        date_match2 = re.search(r'(\d\d.\d\d.)', full_text)
        date = date_match2.group(0) + year
        date_info = parse_date(date)

    #'Day': date_info[0],
    #'Month': date_info[1],
    #'Year': date_info[2],
    
    return {
        'Amount': value_numeric,
        'Date': datetime.strptime(date, '%d.%m.%Y'),
        'Type' : spendingtype,
        'Category': categorize_transaction(full_text),
        'Semester Abroad' : check_semester_abroad(full_text, date),
        'Vacation' : check_vacation(full_text)
    }

# --------------------------
# Main Workflow
# --------------------------
def process_pdf(file_path: str) -> pd.DataFrame:
    """Process a single PDF file"""
    text = read_pdf(file_path)
    year = file_path[17:21]
    _, _, transactions = group_transactions(text)
    return pd.DataFrame([process_transaction(t, year) for t in transactions])

def group_transactions(input_text):
    transactions = []
    kontostand = 0
    kontostand_date = 0
    lines = input_text.split("\n")

    d = re.search(r'(\d+.\d+.\d\d\d\d)', lines[0])
    d = d.group()
    kontostand_date = d
    value = re.search(r'((\d\d\d.|\d\d.|\d.)*\d\d\d,\d\d (H|S))', lines[0])
    value = value.group()
    kontostand = value

    for line in lines:
        if "Übertrag" in line:
            lines.remove(line)

    header_idx = get_header_idx(lines[1:])

    for h in range(len(header_idx)):
        transaction = {}
        idx = header_idx[h]+1
        transaction['header'] = lines[idx]
        if h < len(header_idx)-1:
            next_header = header_idx[h+1]
            transaction['description'] = lines[idx:next_header+1]
        else: 
            transaction['description'] = lines[idx:]
        transactions.append(transaction)

    return kontostand, kontostand_date, transactions

def get_header_idx(lines):
    header_idx = []
    for l in range(0, len(lines)):
        line = lines[l]
        match_value = re.search(r'((\d\d\d,|\d\d,|\d,)\d\d (H|S))', line)
        match_date = re.search(r'(\d\d.\d\d. \d\d.\d\d.)', line)
        if match_value != None and match_date != None:
            header_idx.append(l)
    return header_idx

def get_all_data() -> pd.DataFrame:
    """Parallel PDF processing with ThreadPool"""
    pdf_dir = 'pdfs/'
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
    
    return pd.concat(results, ignore_index=True)

def accumulate_balance(b_df, t_df, start_balance):
    balances = []
    account_balance = start_balance
    for index_b, row_b in b_df.iterrows():
        for index_t, row_t in t_df.iterrows():
            if row_b['Date'] == row_t['Date']:
                account_balance += row_t['Amount']
        balances.append(account_balance)
    return balances

def create_balance_history():
    account_start_balance = 1863.44

    # Load your transaction data (replace with your actual data source)
    transaction_df = pd.read_csv('transactions.csv', sep='\t', encoding='utf-8')  # Should have: Date, Category, Amount, Type
    transaction_df['Date'] = pd.to_datetime(transaction_df['Date'])
    start = transaction_df['Date'].min()
    end = transaction_df['Date'].max()
    pd_start = pd.Timestamp(start)
    pd_end = pd.Timestamp(end)

    transaction_wo_support_df = transaction_df[transaction_df['Category'] != 'Financial Support']

    transaction_df = transaction_df.groupby(['Date']).agg({'Amount':'sum'}).reset_index()
    transaction_wo_support_df = transaction_wo_support_df.groupby(['Date']).agg({'Amount':'sum'}).reset_index()
    dates = pd.date_range(pd_start,pd_end-timedelta(days=1),freq='d')
    balance_df = pd.DataFrame({'Date':dates})

    balance_df['Balance'] = accumulate_balance(balance_df, transaction_df, account_start_balance)
    #balance_df['Balance'] = balance_df['Balance'].apply(lambda x: '{:,.2f}'.format(x))

    balance_df['WithoutSupport'] = accumulate_balance(balance_df, transaction_wo_support_df, account_start_balance)
    #balance_df['WithoutSupport'] = balance_df['WithoutSupport'].apply(lambda x: '{:,.2f}'.format(x))

    balance_df.to_csv('balances.csv', sep='\t', encoding='utf-8', index=False, header=True)

transaction_df = get_all_data()
transaction_df.to_csv('transactions.csv', sep='\t', encoding='utf-8', index=False, header=True)
create_balance_history()