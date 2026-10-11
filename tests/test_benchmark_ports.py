"""Real socket regression tests; does not load a model or start llama-server."""
import importlib.util
from pathlib import Path
import socket
import unittest

spec = importlib.util.spec_from_file_location('gguf_benchmark', Path(__file__).resolve().parents[1] / 'scripts/04_benchmark_gguf.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class PortTests(unittest.TestCase):
    def test_automatic_port_is_bindable(self):
        port = benchmark.select_server_port()
        self.assertGreater(port, 0)
        with socket.socket() as server:
            server.bind(('127.0.0.1', port))
            server.listen(1)

    def test_live_listener_rejected_and_not_disturbed(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            port = server.getsockname()[1]
            server.listen(1)
            with self.assertRaisesRegex(RuntimeError, 'Cannot reserve'):
                benchmark.select_server_port(port)
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                conn, _ = server.accept()
                conn.close()
            self.assertNotEqual(benchmark.select_server_port(), port)

    def test_explicit_port_reusable_after_closed_connection(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 0))
            port = server.getsockname()[1]
            server.listen(1)
            with socket.create_connection(('127.0.0.1', port), timeout=2) as client:
                conn, _ = server.accept()
                # Server actively closes; this can leave its endpoint in TIME_WAIT.
                conn.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b'')
                conn.close()
        try:
            selected = benchmark.select_server_port(port)
        except RuntimeError:
            # Windows exclusive binding can retain TIME_WAIT. Automatic mode
            # must still work, without borrowing another process's socket.
            if not hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                raise
            self.assertGreater(benchmark.select_server_port(), 0)
            return
        self.assertEqual(selected, port)
        with socket.socket() as next_server:
            next_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            next_server.bind(('127.0.0.1', port))
            next_server.listen(1)

    def test_many_sequential_port_reservations(self):
        for _ in range(25):
            self.assertTrue(1 <= benchmark.select_server_port() <= 65535)


if __name__ == '__main__':
    unittest.main(verbosity=2)
