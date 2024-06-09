from whoosh.scoring import WeightLengthScorer, bm25, WeightScorer, BM25F, pl2


class MixedWeighting(BM25F):
    def __init__(self, beta: float = 0.5, c=1.0, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= beta <= 1.0:
            raise ValueError('beta must be between 0 and 1')
        self.beta = beta
        self.c = c

    def scorer(self, searcher, fieldname, text, qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        if fieldname in self._field_B:
            B = self._field_B[fieldname]
        else:
            B = self.B

        return MixedWeightingScorer(searcher, fieldname, text, self.c, B, self.K1, self.beta, qf=qf)


class MixedWeightingScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, c, B, K1, beta, qf=1):
        # IDF and average field length are global statistics, so get them from
        # the top-level searcher
        parent = searcher.get_parent()  # Returns self if no parent
        self.cf = parent.frequency(fieldname, text)
        self.dc = parent.doc_count_all()
        self.idf = parent.idf(fieldname, text)
        self.avgfl = parent.avg_field_length(fieldname) or 1

        self.c = c
        self.B = B
        self.K1 = K1
        self.qf = qf
        self.beta = beta
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        s = self.beta * bm25(self.idf, weight, length, self.avgfl, self.B, self.K1) + (1-self.beta) * pl2(weight, self.cf, self.qf, self.dc, length, self.avgfl,
                   self.c)
        return s
