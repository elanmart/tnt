class HooksContainer:
    def __init__(self):
        self.on_start       = []
        self.on_start_epoch = []
        self.on_sample      = []
        self.on_forward     = []
        self.on_update      = []
        self.on_end_epoch   = []
        self.on_end         = []
        self.on_interrupt   = []
        self.on_failure     = []

        self.active_hooks = set()

        self.register_on_failure(self._raise)
        self.register_on_interrupt(self._raise)

    @property
    def inactive_hooks(self):
        return {name for name in dir(self)
                if name.startswith('on_') and name not in self.active_hooks}

    def register_on_start(self, fn):
        return self._register('on_start', fn)

    def register_on_start_epoch(self, fn):
        return self._register('on_start_epoch', fn)

    def register_on_sample(self, fn):
        return self._register('on_sample', fn)

    def register_on_forward(self, fn):
        return self._register('on_forward', fn)

    def register_on_update(self, fn):
        return self._register('on_update', fn)

    def register_on_end_epoch(self, fn):
        return self._register('on_end_epoch', fn)

    def register_on_end(self, fn):
        return self._register('on_end', fn)

    def register_on_interrupt(self, fn):
        return self._register_singleton('on_interrupt', fn)

    def register_on_failure(self, fn):
        return self._register_singleton('on_failure', fn)

    def register(self, **kwargs):
        for k, v in kwargs.items():
            self._register_singleton(k, v)

    def set(self, fn):
        return self._register_singleton(fn.__name__, fn)

    def _register(self, name, fn):

        if not isinstance(fn, Iterable):
            fn = [fn]

        getattr(self, name).extend(fn)
        self.active_hooks.add(name)

        return fn

    def _register_singleton(self, name, fn):
        getattr(self, name).clear()

        return self._register(name, fn)

    def _raise(self):
        raise

    def __getitem__(self, item):
        return getattr(self, item)


class State:
    def __init__(self, *, forward_fn=None, iterator=None, optimizer=None,
                 epoch=None, maxepoch=None,
                 sample=None, output=None, loss=None,
                 t=None, train=True):

        self.epoch    = epoch
        self.maxepoch = maxepoch

        self.forward_fn = forward_fn
        self.iterator   = iterator
        self.optimizer  = optimizer

        self.sample = sample
        self.output = output
        self.loss   = loss
        self.t      = t
        self.train  = train

    @property
    def test(self):
        return not self.train


# noinspection PyUnresolvedReferences
class Engine:
    def __init__(self, hooks: HooksContainer):
        self._hooks = hooks
        self._ready = False

    def initialize_hooks(self):
        for name in self._hooks.active_hooks:
            self.__setattr__(name, partial(self._run_all, getattr(self._hooks, name)))

        for name in self._hooks.inactive_hooks:
            self.__setattr__(name, self._noop)

        self._ready = True

    def _noop(self, state):
        return

    def _run_all(self, hooks, state):
        for h in hooks:
            h(state)

    def train(self, forward_fn, iterator, optimizer, maxepoch):

        if not self._ready:
            self.initialize_hooks()

        state = State(forward_fn=forward_fn, iterator=iterator, optimizer=optimizer,
                      maxepoch=maxepoch, epoch=0, t=0,
                      train=True)

        self.on_start(state)

        try:
            while state.epoch < state.maxepoch:
                self.on_start_epoch(state)

                for sample in state.iterator:
                    state.sample = sample
                    self.on_sample(state)

                    def closure():
                        loss, output = state.forward_fn(state.sample)

                        state.output = output
                        state.loss = loss
                        loss.backward()

                        self.on_forward(state)

                        state.output = None
                        state.loss = None

                        return loss

                    state.optimizer.zero_grad()
                    state.optimizer.step(closure)

                    self.on_update(state)

                    state.t += 1
                state.epoch += 1

                self.on_end_epoch(state)
            self.on_end(state)

        except KeyboardInterrupt as ex:
            state.exception = ex
            self.on_interrupt(state)

        except Exception as ex:
            state.exception = ex
            self.on_failure(state)

        return state

    def test(self, forward_fn, iterator):

        if not self._ready:
            self.initialize_hooks()

        state = State(forward_fn=forward_fn, iterator=iterator, t=0, train=False)

        self.on_start(state)

        try:
            self.on_start_epoch(state)

            for sample in state.iterator:
                state.sample = sample
                self.on_sample(state)

                def closure():
                    loss, output = state.network(state.sample)

                    state.output = output
                    state.loss = loss

                    self.on_forward(state)

                    state.output = None
                    state.loss = None

                closure()
                state.t += 1

            self.on_end_epoch(state)
            self.on_end(state)

        except KeyboardInterrupt as ex:
            state.exception = ex
            self.on_interrupt(state)

        except Exception as ex:
            state.exception = ex
            self.on_failure(state)

        return state
