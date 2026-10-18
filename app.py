"""Mobile-friendly V29 interface. Prediction and Google Sheets storage unchanged."""

import hashlib
import hmac
from html import escape

import pandas as pd
import streamlit as st

from engine import score_v29, score_details, validate_throw
from storage import open_book, setup, get_state, create_prediction, record_result

st.set_page_config(page_title="V29 Dice Predictor", page_icon="🎲", layout="centered")
st.markdown("""
<style>
.block-container {max-width:700px; padding-top:1.25rem; padding-bottom:2rem;}
div.stButton > button, div.stFormSubmitButton > button {min-height:3.2rem; border-radius:14px; font-size:1rem;}
.pair-grid {display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin:12px 0 20px;}
.pair-card {background:rgba(128,128,128,0.10); border:1px solid rgba(128,128,128,0.24); border-radius:14px; padding:15px 8px; text-align:center;}
.pair-rank {font-size:.78rem; opacity:.72; margin-bottom:4px;}
@media (max-width:650px) {
  .block-container {padding-left:1rem; padding-right:1rem;}
  .pair-grid {grid-template-columns:repeat(3,minmax(0,1fr));}
  .pair-card {padding:16px 5px;}
}
  .pair-value{font-size:1.17rem;}
  div[data-testid="stDataFrame"]{overflow-x:auto;}
}
</style>
""", unsafe_allow_html=True)


def show_pairs(pairs):
    """Display unordered pairs as responsive, read-only cards."""
    cards = "".join(
        '<div class="pair-card"><div class="pair-rank">Rank {}</div>'
        '<div class="pair-value">{}</div></div>'.format(i, escape(str(pair)))
        for i, pair in enumerate(pairs, 1)
    )
    st.markdown('<div class="pair-grid">{}</div>'.format(cards), unsafe_allow_html=True)


st.title("🎲 V29 Dice Predictor")
st.caption("Top-5 unordered pairs · पहला पासा Special (1/2) · Google Sheets")
st.info("मोड: Fixed V28 Prior के साथ V29 continuation. यह Original V28 Generator का प्रमाणित विकल्प नहीं है।")

try:
    required_secrets = ("sheet_id", "gcp_service_account", "app_pin_sha256")
    if any(key not in st.secrets for key in required_secrets):
        st.error("Streamlit Secrets में Google Sheet ID, service account और PIN hash सेट करें।")
        st.stop()

    pin = st.text_input("🔐 ऐप PIN", type="password", help="निजी PIN यहाँ डालें।")
    expected = str(st.secrets["app_pin_sha256"]).lower().strip()
    if not pin or not hmac.compare_digest(hashlib.sha256(pin.encode()).hexdigest(), expected):
        st.info("ऐप खोलने के लिए सही PIN डालें।")
        st.stop()

    book = open_book(st.secrets)
    seed, log, _seed_hash = setup(book)
    history, records, pending = get_state(seed, log)
    st.success(f"Google Sheets connected · {len(history)} saved throws")

    if pending is None:
        prior = seed.get_all_records()[-1]["V28_top5"].split(";")
        preview = score_v29(history, prior)
        details = score_details(history, prior)

        st.subheader(f"अगला Throw #{len(history) + 1}")
        st.caption("यह Preview है। Roll करने से **पहले** Prediction Lock करें।")
        show_pairs(preview)

        with st.expander("📊 V29 Score Dashboard", expanded=False):
            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
            gap = details[4]["V29 Score"] - details[5]["V29 Score"]
            st.caption(f"Rank 5 vs 6 Score Gap: {gap:.6f} (यह जीत की संभावना नहीं है)")

        if st.button("🔒 Prediction Lock & Save", type="primary", use_container_width=True):
            current, _, already = get_state(seed, log)
            if already or current != history:
                st.warning("डेटा दूसरे Session में बदल गया है। पेज Refresh करें।")
            else:
                create_prediction(log, history, preview, prior)
                st.rerun()
    else:
        st.subheader(f"🔒 Locked Throw #{pending['throw']}")
        show_pairs(pending["prediction_top5"].split(";"))
        st.caption(f"Prediction saved: {pending['created_utc']}")
        st.info("अब Dice Roll का वास्तविक Result दर्ज करें। पहला अंक केवल 1 या 2 होगा।")

        with st.form("result_form", clear_on_submit=False):
            result = st.text_input("🎲 Actual Result (जैसे 1,5,6)", placeholder="1,5,6")
            save = st.form_submit_button("✅ Actual Result Save करें", type="primary", use_container_width=True)
            if save:
                try:
                    valid = validate_throw(result)
                    _, _, current_pending = get_state(seed, log)
                    if not current_pending or current_pending["sheet_row"] != pending["sheet_row"]:
                        st.warning("यह Prediction पहले ही पूरी हो चुकी है। Refresh करें।")
                    else:
                        pair, hit = record_result(log, current_pending, valid)
                        st.session_state["last_message"] = (
                            f"Actual pair {pair} · {'HIT ✅' if hit else 'MISS ❌'} · saved"
                        )
                        st.rerun()
                except ValueError as ex:
                    st.error(str(ex))

    if "last_message" in st.session_state:
        st.success(st.session_state.pop("last_message"))

    complete = [r for r in records if r["result"]]
    st.divider()
    st.subheader("📊 Live Accuracy Dashboard")
    if complete:
        hits = sum(int(r["hit"]) for r in complete)
        a, b, c = st.columns(3)
        a.metric("Live Throws", len(complete))
        b.metric("Hits", hits)
        c.metric("Accuracy", f"{100 * hits / len(complete):.2f}%")
        recent = complete[-50:]
        st.caption(f"Last {len(recent)}: {sum(int(r['hit']) for r in recent)}/{len(recent)} hits")
        with st.expander("📋 पुराने Live Results", expanded=False):
            st.dataframe(
                pd.DataFrame([
                    {"Throw": r["throw"], "Top-5": r["prediction_top5"],
                     "Actual": r["result"], "Hit": r["hit"]}
                    for r in reversed(complete)
                ]),
                hide_index=True, use_container_width=True,
            )
    else:
        st.caption("पहला Live Result दर्ज होने पर Dashboard शुरू होगा।")

    st.caption("Data: Historical_Data (672 seed throws) + Live_Log. नई Predictions Result आने से पहले Lock होती हैं।")
except Exception as ex:
    st.error(f"ऐप सेटअप/डेटा जाँच त्रुटि: {ex}")
    st.info("Secrets, Google Sheets और Sheet Tabs की जाँच करें। त्रुटि होने पर Save बंद रहेगा।")
