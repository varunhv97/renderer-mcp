use crate::error::DaemonError;
use std::net::SocketAddr;

/// The daemon holds unauthenticated, unencrypted scene state and accepts
/// requests with no access control beyond "can reach this socket" -- so
/// both [`serve_with_metrics_dir`] (binding) and [`DaemonClient::new`]
/// (connecting) refuse anything but a loopback address, rather than trust
/// callers to only ever pass one.
pub(crate) fn ensure_loopback(endpoint: SocketAddr) -> Result<(), DaemonError> {
    if endpoint.ip().is_loopback() {
        Ok(())
    } else {
        Err(DaemonError::NonLoopbackEndpoint(endpoint))
    }
}

#[cfg(test)]
mod tests {
    use crate::client::DaemonClient;
    use crate::error::DaemonError;
    use crate::protocol::DaemonEnvelope;
    use crate::protocol::DaemonRequest;

    #[test]
    fn validates_protocol_and_loopback_endpoints() {
        assert!(DaemonClient::new("127.0.0.1:9999".parse().unwrap()).is_ok());
        assert!(matches!(
            DaemonClient::new("192.168.1.1:9999".parse().unwrap()),
            Err(DaemonError::NonLoopbackEndpoint(_))
        ));
        let envelope: DaemonEnvelope =
            serde_json::from_str(r#"{"version":"renderer.daemon.v1","method":"health"}"#).unwrap();
        assert!(matches!(envelope.request, DaemonRequest::Health));
        assert!(
            serde_json::from_str::<DaemonEnvelope>(
                r#"{"version":"renderer.daemon.v1","method":"health","extra":true}"#
            )
            .is_err()
        );
    }
}
