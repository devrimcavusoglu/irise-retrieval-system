from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer, SimpleAnalyzer
from whoosh.fields import SchemaClass, TEXT, ID


simple_analyzer = SimpleAnalyzer()  # RegexTokenizer + Lowercase
standard_analyzer = StandardAnalyzer()  # + StopWord filter
irise_analyzer = StemmingAnalyzer()  # + Stemming


class MSMarcoSchema(SchemaClass):
    doc_id = ID(stored=True)
    text = TEXT


class MSMarcoSchemaSimple(MSMarcoSchema):
    text = TEXT(analyzer=simple_analyzer)


class MSMarcoSchemaStandard(MSMarcoSchema):
    text = TEXT(analyzer=standard_analyzer)


class IriseSchema(MSMarcoSchema):
    text = TEXT(analyzer=irise_analyzer)
