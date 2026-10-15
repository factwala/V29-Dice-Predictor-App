"""Google Sheets is source of truth. One mutable pending row per throw; append-only thereafter."""
from datetime import datetime, timezone
import hashlib
import hmac
import json
import gspread
from google.oauth2.service_account import Credentials
from engine import seed_payload, validate_throw

HEADER=['throw','created_utc','model','history_count','history_fingerprint','v28_prior','prediction_top5','result','actual_pair','hit','completed_utc']

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def open_book(settings):
    creds = Credentials.from_service_account_info(dict(settings['gcp_service_account']),scopes=['https://www.googleapis.com/auth/spreadsheets'])
    client = gspread.authorize(creds)
    return client.open_by_key(settings['sheet_id'])

def setup(book):
    data,digest=seed_payload()
    try: seed=book.worksheet('Seed672')
    except gspread.WorksheetNotFound: seed=book.add_worksheet(title='Seed672',rows=750,cols=3)
    existing=seed.get_all_values()
    if not existing:
        seed.update(range_name='A1',values=[['throw','original_result','V28_top5']]+data)
        existing=seed.get_all_values()
    expected=[['throw','original_result','V28_top5']]+data
    if existing!=expected:
        raise RuntimeError('Seed672 differs from original 672 rows. App blocked to protect data integrity.')
    try: log=book.worksheet('LiveLog')
    except gspread.WorksheetNotFound: log=book.add_worksheet(title='LiveLog',rows=5000,cols=len(HEADER))
    vals=log.get_all_values()
    if not vals: log.update(range_name='A1',values=[HEADER])
    elif vals[0]!=HEADER: raise RuntimeError('LiveLog headers do not match the expected format.')
    return seed,log,digest

def get_state(seed,log):
    seedvals=seed.get_all_values()
    history=[validate_throw(row[1]) for row in seedvals[1:]]
    rows=log.get_all_values()[1:]
    records=[]
    pending=None
    for ix,raw in enumerate(rows,2):
        padded=(raw+['']*len(HEADER))[:len(HEADER)]
        rec=dict(zip(HEADER,padded)); rec['sheet_row']=ix
        if int(rec['throw'])!=len(history)+1: raise RuntimeError('LiveLog sequence gap; blocked.')
        # History hash binds the prediction to the preceding observations.
        fingerprint=hashlib.sha256('|'.join(history).encode()).hexdigest()
        if not hmac.compare_digest(rec['history_fingerprint'],fingerprint):
            raise RuntimeError('History fingerprint mismatch. App blocked to protect chronology.')
        if rec['result']:
            result=validate_throw(rec['result'])
            pair=','.join(sorted(result.split(',')[1:],key=int))
            if rec['actual_pair']!=pair or rec['hit']!=str(int(pair in rec['prediction_top5'].split(';'))):
                raise RuntimeError('Recorded result/pair/hit mismatch. App blocked.')
            history.append(result)
        else:
            if ix!=len(rows)+1: raise RuntimeError('Uncompleted throw before another prediction.')
            pending=rec
        records.append(rec)
    return history,records,pending

def create_prediction(log,history,ranking,prior):
    import uuid
    from engine import score_v29
    fingerprint=hashlib.sha256('|'.join(history).encode()).hexdigest()
    top=score_v29(history,prior)
    row=[str(len(history)+1),now(),'V29-continuation-fixed-V28-prior',str(len(history)),fingerprint,';'.join(prior),';'.join(top),'','','','']
    log.append_row(row,value_input_option='RAW')
    return top

def record_result(log,pending,result):
    from engine import unordered_pair
    valid=validate_throw(result); pair=unordered_pair(valid)
    hit=int(pair in pending['prediction_top5'].split(';'))
    # Single update captures result and correct scoring; no later editing offered.
    log.update(range_name=f'H{pending["sheet_row"]}:K{pending["sheet_row"]}',values=[[valid,pair,str(hit),now()]])
    return pair,hit
