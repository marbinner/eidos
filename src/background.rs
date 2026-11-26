//! Work that must not hold up the frame runs here: on a second thread natively, in a
//! Web Worker in the browser. The two sides exchange small byte messages, so the
//! same service code serves both, and can also be run in the frame (or in a test)
//! with neither.
//!
//! A service is addressed by its index in [`crate::scenes::services`], which is the
//! first byte of every message in either direction.

use crate::clock;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// What a service wants after a turn of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Next {
    /// Nothing to do until a message arrives.
    Idle,
    /// Nothing right now, but soon (it is pacing itself).
    Soon,
    /// More work is waiting.
    Now,
}

/// Something that runs in the background.
pub trait Service {
    /// A message from the foreground (without the service byte).
    fn handle(&mut self, msg: &[u8]);
    /// Work for about `seconds`, handing any results to `send`.
    fn work(&mut self, seconds: f64, send: &mut dyn FnMut(Vec<u8>)) -> Next;
}

/// The background's first message: it is up and listening. (Only a Web Worker needs
/// to say so; a thread is up as soon as it exists.)
#[cfg(target_arch = "wasm32")]
const READY: u8 = 255;
/// How long the background works without hearing from the foreground. Whoever owns a
/// piece of background work speaks every frame it draws; a hidden tab or a paused
/// program draws none, and the work stops with it instead of running on unseen.
const PATIENCE: f64 = 0.4;
/// How long the background works before looking for new messages.
const SLICE: f64 = 0.008;

/// The background side: the services, and the rule for when they may work.
pub struct Hub {
    services: Vec<Box<dyn Service>>,
    heard: f64,
}

impl Hub {
    pub fn new() -> Self {
        Hub { services: crate::scenes::services(), heard: f64::NEG_INFINITY }
    }

    pub fn receive(&mut self, msg: &[u8]) {
        self.heard = clock::now();
        if let Some((&id, rest)) = msg.split_first() {
            if let Some(service) = self.services.get_mut(id as usize) {
                service.handle(rest);
            }
        }
    }

    pub fn turn(&mut self, send: &mut dyn FnMut(Vec<u8>)) -> Next {
        if clock::now() - self.heard > PATIENCE {
            return Next::Idle;
        }
        let mut next = Next::Idle;
        for (id, service) in self.services.iter_mut().enumerate() {
            let wants = service.work(SLICE, &mut |mut msg| {
                msg.insert(0, id as u8);
                send(msg);
            });
            next = next.max(wants);
        }
        next
    }
}

impl Default for Hub {
    fn default() -> Self {
        Hub::new()
    }
}

/// The foreground's end of the connection.
pub struct Link {
    channel: imp::Channel,
    inbox: RefCell<VecDeque<Vec<u8>>>,
}

impl Link {
    /// Is the background up? Until it is, nothing may be sent (and the caller waits
    /// a moment, or does the work itself).
    pub fn ready(&self) -> bool {
        self.channel.state() == State::Ready
    }

    /// Did it fail to start, or stop since? Then it will never answer again: whoever
    /// had work there takes it back.
    pub fn failed(&self) -> bool {
        self.channel.state() == State::Failed
    }

    pub fn send(&self, service: u8, payload: &[u8]) {
        let mut msg = Vec::with_capacity(payload.len() + 1);
        msg.push(service);
        msg.extend_from_slice(payload);
        self.channel.post(msg);
    }

    /// Hand every waiting message of `service` that `mine` accepts to `take`, in
    /// order of arrival. Messages nobody claims are dropped once enough newer ones
    /// have arrived.
    pub fn take(&self, service: u8, mut mine: impl FnMut(&[u8]) -> bool, mut take: impl FnMut(&[u8])) {
        let mut inbox = self.inbox.borrow_mut();
        self.channel.poll(&mut inbox);
        inbox.retain(|msg| match msg.split_first() {
            Some((&id, payload)) if id == service && mine(payload) => {
                take(payload);
                false
            }
            _ => true,
        });
        while inbox.len() > 64 {
            inbox.pop_front();
        }
    }
}

thread_local! {
    static LINK: RefCell<Option<Rc<Link>>> = const { RefCell::new(None) };
    /// Set when the background could not be started at all, so that it is tried once
    /// and not again on every frame.
    static NONE_TO_BE_HAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The connection to the background, started on first use. `None` where background
/// work is not wanted: in reproducible runs, whose results must not depend on timing.
pub fn link() -> Option<Rc<Link>> {
    if clock::reproducible() {
        return None;
    }
    LINK.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() && !NONE_TO_BE_HAD.get() {
            match imp::spawn() {
                Some(channel) => *slot = Some(Rc::new(Link { channel, inbox: RefCell::new(VecDeque::new()) })),
                None => NONE_TO_BE_HAD.set(true),
            }
        }
        slot.clone()
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    #[cfg(target_arch = "wasm32")]
    Starting,
    Ready,
    Failed,
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use super::{Hub, Next, State};
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::sync::mpsc::{self, RecvTimeoutError, TryRecvError};
    use std::time::Duration;

    pub struct Channel {
        to_background: mpsc::Sender<Vec<u8>>,
        from_background: mpsc::Receiver<Vec<u8>>,
        /// Set once a message could not be handed over: the thread has ended (a
        /// panic in a service, where panics unwind).
        gone: Cell<bool>,
    }

    /// A thread that works while there is work, naps while a service is pacing
    /// itself, sleeps until spoken to otherwise, and ends when the link is dropped.
    pub fn spawn() -> Option<Channel> {
        let (to_background, inbox) = mpsc::channel::<Vec<u8>>();
        let (outbox, from_background) = mpsc::channel::<Vec<u8>>();
        std::thread::Builder::new()
            .name("eidos background".into())
            .spawn(move || {
                let mut hub = Hub::new();
                loop {
                    let next = hub.turn(&mut |msg| {
                        let _ = outbox.send(msg);
                    });
                    let first = match next {
                        Next::Now => match inbox.try_recv() {
                            Ok(msg) => Some(msg),
                            Err(TryRecvError::Empty) => None,
                            Err(TryRecvError::Disconnected) => return,
                        },
                        Next::Soon => match inbox.recv_timeout(Duration::from_millis(1)) {
                            Ok(msg) => Some(msg),
                            Err(RecvTimeoutError::Timeout) => None,
                            Err(RecvTimeoutError::Disconnected) => return,
                        },
                        Next::Idle => match inbox.recv() {
                            Ok(msg) => Some(msg),
                            Err(_) => return,
                        },
                    };
                    if let Some(msg) = first {
                        hub.receive(&msg);
                        while let Ok(msg) = inbox.try_recv() {
                            hub.receive(&msg);
                        }
                    }
                }
            })
            .ok()?;
        Some(Channel { to_background, from_background, gone: Cell::new(false) })
    }

    impl Channel {
        pub fn state(&self) -> State {
            if self.gone.get() {
                State::Failed
            } else {
                State::Ready
            }
        }

        pub fn post(&self, msg: Vec<u8>) {
            if self.to_background.send(msg).is_err() && !self.gone.replace(true) {
                log::warn!("the background thread has ended; its work will be done in the frame instead");
            }
        }

        pub fn poll(&self, inbox: &mut VecDeque<Vec<u8>>) {
            while let Ok(msg) = self.from_background.try_recv() {
                inbox.push_back(msg);
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    use super::{Hub, Next, State, READY};
    use js_sys::{Array, Uint8Array};
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use wasm_bindgen::prelude::*;
    use web_sys::{DedicatedWorkerGlobalScope, ErrorEvent, MessageChannel, MessageEvent, Worker, WorkerOptions, WorkerType};

    pub struct Channel {
        worker: Worker,
        state: Rc<Cell<State>>,
        arrived: Rc<RefCell<VecDeque<Vec<u8>>>>,
        _on_message: Closure<dyn FnMut(MessageEvent)>,
        _on_error: Closure<dyn FnMut(ErrorEvent)>,
    }

    /// A Web Worker running a second copy of this program (see `web/worker.js`),
    /// which calls [`worker_start`] instead of opening a window. The page hands the
    /// worker its own compiled program as the first message: the worker then runs
    /// exactly the build the page runs (the two must agree on every message), and
    /// neither downloads nor compiles it a second time.
    pub fn spawn() -> Option<Channel> {
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        let worker = Worker::new_with_options("./worker.js", &options).ok()?;
        let state = Rc::new(Cell::new(State::Starting));
        if worker.post_message(&wasm_bindgen::module()).is_err() {
            worker.terminate();
            state.set(State::Failed);
        }
        let arrived = Rc::new(RefCell::new(VecDeque::new()));
        let on_message = {
            let (state, arrived) = (state.clone(), arrived.clone());
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                let bytes = Uint8Array::new(&event.data()).to_vec();
                if bytes == [READY] {
                    state.set(State::Ready);
                } else {
                    arrived.borrow_mut().push_back(bytes);
                }
            })
        };
        worker.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        let on_error = {
            let (state, worker) = (state.clone(), worker.clone());
            Closure::<dyn FnMut(ErrorEvent)>::new(move |event: ErrorEvent| {
                // The worker is of no more use (after a panic it cannot even take a
                // message), so it is stopped, and whoever had work there carries on
                // in the frame from the last state it sent (see `Link::failed`). The
                // event is ours to handle: left alone, the browser would report it
                // on the page as an error of the page's own.
                event.prevent_default();
                if state.replace(State::Failed) != State::Failed {
                    log::warn!("the background worker failed ({}); its work will be done in the frame instead", event.message());
                    worker.terminate();
                }
            })
        };
        worker.set_onerror(Some(on_error.as_ref().unchecked_ref()));
        Some(Channel { worker, state, arrived, _on_message: on_message, _on_error: on_error })
    }

    impl Channel {
        pub fn state(&self) -> State {
            self.state.get()
        }

        pub fn post(&self, msg: Vec<u8>) {
            let buffer = Uint8Array::from(&msg[..]).buffer();
            let _ = self.worker.post_message_with_transfer(&buffer, &Array::of1(&buffer));
        }

        pub fn poll(&self, inbox: &mut VecDeque<Vec<u8>>) {
            inbox.extend(self.arrived.borrow_mut().drain(..));
        }
    }

    impl Drop for Channel {
        fn drop(&mut self) {
            self.worker.terminate();
        }
    }

    /// The worker's side: called once by `web/worker.js` after the program has loaded
    /// there. It works in short slices and returns to the browser between them, so
    /// that messages from the page are heard.
    #[wasm_bindgen]
    pub fn worker_start() {
        // (so that a panic here says what it was before the worker stops)
        eframe::WebLogger::init(log::LevelFilter::Info).ok();
        std::panic::set_hook(Box::new(|info| log::error!("the background worker panicked: {info}")));
        let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
        let hub = Rc::new(RefCell::new(Hub::new()));
        // A message to ourselves is the one way to say "call me back at once": timers
        // are held back to a few milliseconds.
        let Ok(wake) = MessageChannel::new() else { return };
        let scheduled = Rc::new(Cell::new(false));
        let call_back = {
            let (port, scheduled) = (wake.port2(), scheduled.clone());
            Rc::new(move || {
                if !scheduled.replace(true) {
                    let _ = port.post_message(&JsValue::NULL);
                }
            })
        };
        let after_a_nap = {
            let port = wake.port2();
            Closure::<dyn FnMut()>::new(move || {
                let _ = port.post_message(&JsValue::NULL);
            })
        };
        let turn = {
            let (hub, scope, scheduled, call_back) = (hub.clone(), scope.clone(), scheduled.clone(), call_back.clone());
            Closure::<dyn FnMut(MessageEvent)>::new(move |_: MessageEvent| {
                scheduled.set(false);
                let next = hub.borrow_mut().turn(&mut |msg| {
                    let buffer = Uint8Array::from(&msg[..]).buffer();
                    let _ = scope.post_message_with_transfer(&buffer, &Array::of1(&buffer));
                });
                match next {
                    Next::Now => call_back(),
                    Next::Soon => {
                        if !scheduled.replace(true) {
                            let _ = scope.set_timeout_with_callback_and_timeout_and_arguments_0(after_a_nap.as_ref().unchecked_ref(), 1);
                        }
                    }
                    Next::Idle => {}
                }
            })
        };
        wake.port1().set_onmessage(Some(turn.as_ref().unchecked_ref()));
        let on_message = {
            let hub = hub.clone();
            Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
                hub.borrow_mut().receive(&Uint8Array::new(&event.data()).to_vec());
                call_back();
            })
        };
        scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
        // these live as long as the worker does
        turn.forget();
        on_message.forget();
        let _ = scope.post_message(&Uint8Array::from(&[READY][..]).buffer());
    }
}
