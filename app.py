import hashlib
import hmac
import streamlit as st
import pandas as pd
from engine import score_v29, validate_throw
from storage import open_book, setup, get_state, create_prediction, record_result

st.set_page_config(page_title='V29 Dice Predictor', page_icon='🎲', layout='centered')
st.markdown('''<style> .block-container{max-width:650px;padding-top:1.5rem} div.stButton>button{width:100%;border-radius:13px;min-height:2.7rem} </style>''',unsafe_allow_html=True)
st.title('🎲 V29 Dice Predictor')
st.caption('Top-5 unordered pair • पहला पासा special (1/2) • Google Sheets history')
st.info('मोड: V29 continuation (अंतिम उपलब्ध V28 ranking को fixed रखा गया है)। यह नए throw के लिए original V28 generator का प्रमाणित विकल्प नहीं है।')
try:
    if 'sheet_id' not in st.secrets or 'gcp_service_account' not in st.secrets or 'app_pin_sha256' not in st.secrets:
        st.error('पहले Streamlit Secrets में Google Sheet ID, service account और app PIN hash सेट करें। README देखें।');st.stop()
    pin=st.text_input('🔐 ऐप PIN',type='password',help='Private access; shared app URL से data access रोकने के लिए।')
    expected=str(st.secrets['app_pin_sha256']).lower().strip()
    if not pin or not hmac.compare_digest(hashlib.sha256(pin.encode()).hexdigest(),expected):
        st.info('ऐप खोलने के लिए सही PIN डालें।');st.stop()
    book=open_book(st.secrets)
    seed,log,seed_hash=setup(book)
    history, records,pending=get_state(seed,log)
    st.success(f'Google Sheets connected • {len(history)} saved throws')
    if not pending:
        prior=seed.get_all_records()[-1]['V28_top5'].split(';')
        st.subheader(f'अगला Throw #{len(history)+1}')
        preview=score_v29(history,prior)
        st.write(' '.join(f'**{j}. `{p}`**　' for j,p in enumerate(preview,1)))
        st.caption('यह preview है; नीचे LOCK दबाने पर prediction Google Sheets में दर्ज होगी।')
        if st.button('🔒 Prediction Lock & Save',type='primary'):
            # Refresh in case another phone/tab changed the sheet since page load.
            current,_,already=get_state(seed,log)
            if already or current!=history: st.warning('दूसरे session में डेटा बदला है। Refresh करें।')
            else:
                create_prediction(log,history,preview,prior)
                st.rerun()
    else:
        st.subheader(f'🔒 Locked Throw #{pending["throw"]}')
        top=pending['prediction_top5'].split(';')
        for j,p in enumerate(top,1): st.write(f'{j}.  **{p}**')
        st.caption(f'Prediction saved at {pending["created_utc"]}. अब परिणाम भरें।')
        with st.form('result_form',clear_on_submit=False):
            result=st.text_input('वास्तविक तीन पासे (जैसे 1,5,6)',placeholder='1,5,6')
            save=st.form_submit_button('✅ Actual Result सेव करें',type='primary')
            if save:
                try:
                    valid=validate_throw(result)
                    _,_,current_pending=get_state(seed,log)
                    if not current_pending or current_pending['sheet_row']!=pending['sheet_row']:
                        st.warning('यह prediction दूसरे session में पूरी हो गई है; Refresh करें।')
                    else:
                        pair,hit=record_result(log,current_pending,valid)
                        st.session_state['last_message']=f'Actual pair {pair} • {"HIT ✅" if hit else "MISS ❌"} • saved'
                        st.rerun()
                except ValueError as ex: st.error(str(ex))
    if 'last_message' in st.session_state: st.success(st.session_state.pop('last_message'))
    complete=[r for r in records if r['result']]
    st.divider()
    st.subheader('📊 Live Accuracy Dashboard')
    if complete:
        hits=sum(int(r['hit']) for r in complete)
        a,b,c=st.columns(3)
        a.metric('Live throws',len(complete));b.metric('Hits',hits);c.metric('Accuracy',f'{100*hits/len(complete):.2f}%')
        recent=complete[-50:]; st.caption(f'Last {len(recent)}: {sum(int(r["hit"]) for r in recent)}/{len(recent)} hits')
        st.dataframe(pd.DataFrame([{'Throw':r['throw'],'Top-5':r['prediction_top5'],'Actual':r['result'],'Hit':r['hit']} for r in reversed(complete)]),hide_index=True,use_container_width=True)
    else: st.caption('नया लाइव परिणाम दर्ज होते ही dashboard शुरू होगा। Historical 672 predictions को live accuracy में नहीं जोड़ा जाता।')
    st.caption('Data source: Google Sheet Seed672 + LiveLog. फाइल में 672 seed परिणाम केवल history के लिए हैं; नई predictions results से पहले लॉक होती हैं।')
except Exception as ex:
    st.error(f'ऐप सेटअप/डेटा जाँच त्रुटि: {ex}')
    st.info('Secrets, Google Sheets sharing और Sheet की tabs जाँचें। सुरक्षा के लिए त्रुटि होने पर prediction/save बंद रहेगा।')
