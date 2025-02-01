import zmq
import msgpack
import os
import time
import threading
import multiprocessing
from contextlib import contextmanager

class TraceLogger:
    def __init__(self, script_name, process_name=None):
        self.script_name = script_name
        self.process_name = process_name or script_name  # Default to script_name if not provided
        self.pid = os.getpid()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.ipc_path = os.path.join(os.getcwd(), ".trace_socket")
        self.socket.connect(f"ipc://{self.ipc_path}")
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self._event_starts = {}
        
        # Send process metadata event
        self._send_process_metadata()
        
    def _send_process_metadata(self):
        """Send process metadata event to help identify processes in the trace."""
        metadata = {
            "name": "process_name",
            "ph": "M",  # Metadata event
            "pid": self.pid,
            "tid": 0,  # Metadata events use tid 0 by convention
            "args": {"name": self.process_name}
        }
        try:
            self.socket.send(msgpack.packb(metadata), zmq.NOBLOCK)
        except (zmq.Again, Exception) as e:
            print(f"Warning: Could not send process metadata: {e}")
        
    def _get_thread_id(self):
        if hasattr(threading, 'get_native_id'):
            return threading.get_native_id()
        return threading.current_thread().ident

    def startEvent(self, event_name):
        """Start timing an event."""
        tid = self._get_thread_id()
        current_ts = int(time.time() * 1_000_000)
        key = (event_name, tid)
        self._event_starts[key] = current_ts  # Record start time in microseconds

    def stopEvent(self, event_name):
        """Stop timing an event and send the completed event with duration."""
        tid = self._get_thread_id()
        key = (event_name, tid)
        
        if key not in self._event_starts:
            print(f"Warning: No start time found for event {event_name}")
            return
            
        start_time = self._event_starts.pop(key)
        end_time = time.time() * 1_000_000  # End time in microseconds
        duration = end_time - start_time
        
        event = {
            "name": event_name,
            "ph": "X",
            "ts": int(start_time),
            "dur": int(duration),
            "pid": self.pid,
            "tid": tid
        }
        try:
            self.socket.send(msgpack.packb(event), zmq.NOBLOCK)
        except zmq.Again:
            print("Warning: Trace event dropped due to full queue")
        except Exception as e:
            print(f"Error sending trace event: {e}")
            
    def instantEvent(self, event_name, args=None):
        """Log an instant event with optional arguments."""
        current_ts = int(time.time() * 1_000_000)
        event = {
            "name": event_name,
            "ph": "i",  # Instant event
            "ts": current_ts,
            "pid": self.pid,
            "tid": self._get_thread_id(),
            "s": "g"  # Global scope
        }
        if args:
            event["args"] = args
        try:
            self.socket.send(msgpack.packb(event), zmq.NOBLOCK)
        except zmq.Again:
            print("Warning: Instant trace event dropped due to full queue")
        except Exception as e:
            print(f"Error sending instant trace event: {e}")
            
    def __del__(self):
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.destroy()

    @contextmanager
    def event_scope(self, event_name):
        """Context manager for timing events. Usage:
        with logger.event_scope("my_event"):
            # code to time
        """
        try:
            self.startEvent(event_name)
            yield
        finally:
            self.stopEvent(event_name)