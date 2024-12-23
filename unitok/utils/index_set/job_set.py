from unitok.job import Job
from unitok.tokenizer.union_tokenizer import UnionTokenizer
from unitok.utils.index_set.index_set import IndexSet


class JobSet(IndexSet[Job]):
    @staticmethod
    def _get_key(obj: Job):
        return obj.name

    def next_order(self):
        return max([job.order for job in self]) + 1

    def merge(self, other: IndexSet[Job], **kwargs):
        key_job = kwargs.get('key_job')

        next_order = self.next_order()
        for job in other:
            if job is key_job:
                continue
            if not job.is_processed:
                raise ValueError(f'Merge unprocessed job: {job}')
            if self.has(self._get_key(job)):
                raise ValueError(f'Conflict job name: {job.name}')
            self.add(job.clone(order=next_order, tokenizer=UnionTokenizer(job.tokenizer)))
