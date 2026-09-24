// Minimal request/response RPC over postMessage, used between the page and
// the two kinds of worker (SQLite, encoder).
//
//   worker side:  serve(self, { async method(args, notify) { … return result; } })
//   page side:    const call = client(worker); await call('method', args, onNotify)
//
// A handler may call notify(payload) any number of times (progress events);
// the page receives them through the onNotify callback of that call.
// Handlers may return {__transfer: [buffers], value} to transfer buffers.

export function serve(port, handlers, { serial = false, priority = null } = {}) {
  // serial: run one handler at a time (handlers are async and would otherwise
  // interleave at every await), in arrival order within a priority level;
  // priority(method) -> number, lower runs first (default 0 for every call).
  // A running handler is never interrupted: priorities only reorder the queue.
  const queue = [];
  let busy = false;
  const pump = async () => {
    if (busy) return;
    busy = true;
    while (queue.length) {
      let best = 0;
      for (let i = 1; i < queue.length; i++) if (queue[i].prio < queue[best].prio) best = i;
      const [job] = queue.splice(best, 1);
      try { await handle(job.e); } catch { /* handle() reports errors to the caller */ }
    }
    busy = false;
  };
  port.onmessage = (e) => {
    if (!serial) return handle(e);
    queue.push({ e, prio: priority ? priority(e.data && e.data.method) : 0 });
    return pump();
  };
  const handle = async (e) => {
    const { id, method, args } = e.data || {};
    if (id === undefined) return;
    const notify = (payload) => port.postMessage({ id, notify: payload });
    try {
      const h = handlers[method];
      if (!h) throw new Error(`unknown method ${method}`);
      const { result, transfer } = wrap(await h(args || {}, notify));
      port.postMessage({ id, result }, transfer);
    } catch (err) {
      port.postMessage({ id, error: String(err && err.message || err), stack: err && err.stack });
    }
  };
}

function wrap(r) {
  if (r && r.__transfer) return { result: r.value, transfer: r.__transfer };
  return { result: r, transfer: [] };
}

export function client(worker) {
  let next = 1;
  const pending = new Map();
  worker.onmessage = (e) => {
    const { id, result, error, notify } = e.data || {};
    const p = pending.get(id);
    if (!p) return;
    if (notify !== undefined) { p.onNotify?.(notify); return; }
    pending.delete(id);
    if (error !== undefined) p.reject(new Error(error));
    else p.resolve(result);
  };
  worker.onerror = (e) => {
    const err = new Error(`worker error: ${e.message || e}`);
    for (const p of pending.values()) p.reject(err);
    pending.clear();
  };
  return (method, args, onNotify, transfer = []) => new Promise((resolve, reject) => {
    const id = next++;
    pending.set(id, { resolve, reject, onNotify });
    worker.postMessage({ id, method, args }, transfer);
  });
}
