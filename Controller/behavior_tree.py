from enum import Enum, auto


class NodeStatus(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BehaviorNode:
    def tick(self):
        raise NotImplementedError


class Condition(BehaviorNode):
    def __init__(self, predicate):
        self.predicate = predicate

    def tick(self):
        return NodeStatus.SUCCESS if self.predicate() else NodeStatus.FAILURE


class Action(BehaviorNode):
    def __init__(self, action):
        self.action = action

    def tick(self):
        return self.action()


class Sequence(BehaviorNode):
    def __init__(self, *children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status is not NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class Selector(BehaviorNode):
    def __init__(self, *children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status is not NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE
