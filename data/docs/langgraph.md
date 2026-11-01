# LangGraph

## StateGraph

LangGraph represents a workflow as nodes that read and update shared state.
`StateGraph` is commonly parameterized by a `TypedDict` that describes the
values passed between nodes.

```python
from typing import TypedDict
from langgraph.graph import END, START, StateGraph

class State(TypedDict):
    question: str
    answer: str

def answer(state: State) -> dict[str, str]:
    return {"answer": f"You asked: {state['question']}"}

builder = StateGraph(State)
builder.add_node("answer", answer)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)
app = builder.compile()
```

Invoke a compiled graph with an initial state dictionary. Nodes return only the
state keys they update.

## Conditional edges

`add_conditional_edges` chooses a next node using a routing function. The
function's return value maps to a destination node. This supports branching,
bounded retry loops, and explicit fallback paths.

```python
def route(state: State) -> str:
    return "answer" if state["question"] else "no_answer"

builder.add_conditional_edges(
    "grade",
    route,
    {"answer": "answer", "no_answer": "no_answer"},
)
```

## Graph termination

Connect terminal nodes to `END`. A retry loop must track an attempt count in
state and route to a terminal node when its configured limit is reached.
