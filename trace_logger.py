import zmq
import msgpack
import os
import time
import threading
import multiprocessing

class TraceLogger:
    def __init__(self, script_name, process_name=None):
        self.script_name = script_name
        self.process_name = process_name or script_name  # Default to script_name if no process_name given
        self.pid = os.getpid()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.ipc_path = os.path.join(os.getcwd(), ".trace_socket")
        self.socket.connect(f"ipc://{self.ipc_path}")
        self.socket.setsockopt(zmq.SNDTIMEO, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        # Keep track of start times for events
        self._event_starts = {}
        
        # Send process name metadata event
        self._send_process_metadata()
        
    def _send_process_metadata(self):
        """Send process metadata event to help identify processes in the trace"""
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
        # Try to get native thread id first
        if hasattr(threading, 'get_native_id'):
            return threading.get_native_id()
        # Fallback to thread ident
        return threading.current_thread().ident
        
    def startEvent(self, event_name):
        """Start timing an event"""
        key = (event_name, self._get_thread_id())
        self._event_starts[key] = time.time() * 1_000_000  # Convert to microseconds
        
    def stopEvent(self, event_name):
        """Stop timing an event and send the trace"""
        tid = self._get_thread_id()
        key = (event_name, tid)
        
        if key not in self._event_starts:
            print(f"Warning: No start time found for event {event_name}")
            return
            
        start_time = self._event_starts.pop(key)
        end_time = time.time() * 1_000_000  # Convert to microseconds
        duration = end_time - start_time
        
        # Construct the trace event
        event = {
            "name": event_name,  # Remove script_name prefix
            "ph": "X",
            "ts": int(start_time),
            "dur": int(duration),
            "pid": self.pid,
            "tid": tid
        }
        
        try:
            # Send the event
            self.socket.send(msgpack.packb(event), zmq.NOBLOCK)
        except zmq.Again:
            print(f"Warning: Trace event dropped due to full queue")
        except Exception as e:
            print(f"Error sending trace: {e}")
            
    def __del__(self):
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.destroy() 