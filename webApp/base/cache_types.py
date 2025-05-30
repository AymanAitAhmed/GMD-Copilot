from enum import Enum


class CacheTypes(Enum):
    QUESTION = "question"
    SQL = "sql"
    PLOTLY_CODE = "plotly_code"
    DF = "df"
    FIG_JSON = "fig_json"
    SPEC_JSON = "spec_json"
    SUMMARY_ANSWER = "summary_answer"
    STATUS = 'status'
    