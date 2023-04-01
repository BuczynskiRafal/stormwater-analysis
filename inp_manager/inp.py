import swmmio as sw
from swmmio.utils.functions import trace_from_node

from main import model


class SwmmModel:

    def __init__(self, model: sw.Model) -> None:
        self.model = model

    def find_all_traces(self):
        outfalls = self.model.inp.outfalls
        traces = []
        for outfall in outfalls.index:
            trace_result = trace_from_node(self.model.inp.conduits, outfall)
            traces.append(trace_result)
        return traces

o = SwmmModel(model)
print(o.find_all_traces())
