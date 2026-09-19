use crate::endpoint::ensure_loopback;
use crate::error::DaemonError;
use crate::protocol::{
    DAEMON_PROTOCOL_VERSION, DaemonEnvelope, DaemonRequest, DaemonResponse, DaemonResult,
    MAX_REQUEST_BYTES,
};
#[cfg(doc)]
use crate::server::serve;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::Duration;

const CLIENT_IO_TIMEOUT: Duration = Duration::from_secs(5);

/// A TCP client for a running [`serve`]d daemon: one request/response round
/// trip per [`DaemonClient::call`], each on its own freshly connected
/// stream (no persistent connection/session to manage).
#[derive(Clone, Debug)]
pub struct DaemonClient {
    endpoint: SocketAddr,
    timeout: Duration,
}

impl DaemonClient {
    pub fn new(endpoint: SocketAddr) -> Result<Self, DaemonError> {
        ensure_loopback(endpoint)?;
        Ok(Self {
            endpoint,
            timeout: CLIENT_IO_TIMEOUT,
        })
    }

    pub fn call(&self, request: DaemonRequest) -> Result<DaemonResult, DaemonError> {
        let mut stream = TcpStream::connect_timeout(&self.endpoint, self.timeout)
            .map_err(DaemonError::Connection)?;
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(DaemonError::Connection)?;
        stream
            .set_write_timeout(Some(self.timeout))
            .map_err(DaemonError::Connection)?;
        serde_json::to_writer(
            &mut stream,
            &DaemonEnvelope {
                version: DAEMON_PROTOCOL_VERSION.into(),
                request,
            },
        )
        .map_err(|error| DaemonError::Protocol(error.to_string()))?;
        stream.write_all(b"\n").map_err(DaemonError::Connection)?;
        stream.flush().map_err(DaemonError::Connection)?;
        let mut bytes = Vec::with_capacity(1024);
        BufReader::new(stream)
            .take((MAX_REQUEST_BYTES + 1) as u64)
            .read_until(b'\n', &mut bytes)
            .map_err(DaemonError::Connection)?;
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(DaemonError::Protocol("response exceeds 1 MiB limit".into()));
        }
        let response: DaemonResponse = serde_json::from_slice(&bytes)
            .map_err(|error| DaemonError::Protocol(error.to_string()))?;
        if response.version != DAEMON_PROTOCOL_VERSION {
            return Err(DaemonError::Protocol(
                "unsupported daemon response version".into(),
            ));
        }
        response.result.ok_or_else(|| match response.error {
            Some(error) => DaemonError::Remote {
                code: error.code,
                message: error.message,
            },
            None => DaemonError::Protocol("daemon returned no result".into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::BufRead;
    use std::io::BufReader;
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn client_times_out_when_a_peer_never_responds() {
        let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
            return;
        };
        let endpoint = listener.local_addr().unwrap();
        let peer = thread::spawn(move || {
            let _stream = listener.accept().unwrap().0;
            thread::sleep(Duration::from_millis(50));
        });
        let client = DaemonClient {
            endpoint,
            timeout: Duration::from_millis(10),
        };
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Connection(error)) if matches!(error.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock)
        ));
        peer.join().unwrap();
    }

    #[test]
    fn client_call_reports_oversized_or_malformed_responses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(&vec![b'a'; MAX_REQUEST_BYTES + 2]);
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message.contains("exceeds 1 MiB limit")
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ =
                stream.write_all(br#"{"version":"nope","result":{"kind":"health"},"error":null}"#);
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message.contains("unsupported daemon response version")
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ =
                stream.write_all(br#"{"version":"renderer.daemon.v1","result":null,"error":null}"#);
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Protocol(message)) if message == "daemon returned no result"
        ));
        server.join().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(
                br#"{"version":"renderer.daemon.v1","result":null,"error":{"code":"not_found","message":"missing"}}"#,
            );
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        assert!(matches!(
            client.call(DaemonRequest::Health),
            Err(DaemonError::Remote { code, message })
                if code == "not_found" && message == "missing"
        ));
        server.join().unwrap();
    }

    #[test]
    fn client_call_surfaces_remote_errors_with_their_code_and_message() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let _ = BufReader::new(&stream)
                .take(MAX_REQUEST_BYTES as u64)
                .read_until(b'\n', &mut request);
            let _ = stream.write_all(
                br#"{"version":"renderer.daemon.v1","result":null,"error":{"code":"revision_conflict","message":"scene revision conflict: expected 1, current 2"}}"#,
            );
            let _ = stream.write_all(b"\n");
        });
        let client = DaemonClient::new(endpoint).unwrap();
        let error = client.call(DaemonRequest::Health).unwrap_err();
        assert_eq!(error.code(), "revision_conflict");
        assert!(matches!(
            error,
            DaemonError::Remote { code, message }
                if code == "revision_conflict"
                    && message == "scene revision conflict: expected 1, current 2"
        ));
        server.join().unwrap();
    }
}
