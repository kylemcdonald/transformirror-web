import zmq
import msgpack
import json
import signal
import os
import atexit
import sys

class TraceReceiver:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.ipc_path = os.path.join(os.getcwd(), ".trace_socket")
        self.socket.bind(f"ipc://{self.ipc_path}")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.trace_events = []
        self.running = True
        
        # Register cleanup handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        atexit.register(self.save_traces)
        
    def cleanup(self, *args):
        self.running = False
        self.save_traces()
        self.socket.close()
        self.context.destroy()
        sys.exit(0)
        
    def save_traces(self):
        if self.trace_events:
            trace_data = {
                "traceEvents": self.trace_events
            }
            with open("perfetto_trace.json", "w") as f:
                json.dump(trace_data, f, indent=2)
            print(f"Saved {len(self.trace_events)} trace events to perfetto_trace.json")
            
    def run(self):
        print("Trace receiver started. Press Ctrl+C to save and exit.")
        while self.running:
            try:
                msg = self.socket.recv()
                event = msgpack.unpackb(msg)
                self.trace_events.append(event)
            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error receiving trace: {e}")

if __name__ == "__main__":
    receiver = TraceReceiver()
    receiver.run() 