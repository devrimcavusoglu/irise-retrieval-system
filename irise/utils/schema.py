from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
from whoosh.fields import SchemaClass, TEXT, ID


irise_analyzer = RegexTokenizer() | LowercaseFilter() | StopFilter()


class MSMarcoSchema(SchemaClass):
    doc_id = ID(stored=True)
    text = TEXT


class IriseSchema(MSMarcoSchema):
    text = TEXT(analyzer=irise_analyzer)
