//! 熔断器实现
//!
//! 基于 Circuit Breaker 模式，提供服务超时和故障保护机制。
//!
//! # 状态机
//!
//! ```text
//! Closed (正常) --[失败次数达到阈值]--> Open (熔断)
//!       ^                                   |
//!       |                                   v
//!       |                            Half-Open (半开)
//!       |                                   |
//!       +-------[成功]----------------------+
//! ```

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// 熔断器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// 正常状态，允许请求通过
    Closed,
    /// 熔断状态，拒绝所有请求
    Open,
    /// 半开状态，允许一个请求尝试恢复
    HalfOpen,
}

/// 熔断器配置
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// 失败次数阈值（达到后触发熔断）
    pub failure_threshold: u32,
    /// 熔断持续时间（秒）
    pub open_duration_secs: u64,
    /// 半开状态允许的最大尝试次数
    pub half_open_max_attempts: u32,
    /// 请求超时时间（毫秒）
    pub timeout_ms: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_duration_secs: 30,
            half_open_max_attempts: 1,
            timeout_ms: 5000,
        }
    }
}

/// 熔断器内部状态
#[derive(Debug)]
struct CircuitBreakerInner {
    /// 当前状态
    state: CircuitState,
    /// 连续失败次数
    failure_count: u32,
    /// 连续成功次数（用于半开状态）
    success_count: u32,
    /// 最后失败时间
    last_failure_time: Option<Instant>,
    /// 半开状态尝试次数
    half_open_attempts: u32,
}

/// 熔断器
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    inner: Arc<RwLock<CircuitBreakerInner>>,
    config: CircuitBreakerConfig,
    /// 熔断开始时间
    open_time: Arc<RwLock<Option<Instant>>>,
}

impl CircuitBreaker {
    /// 创建新的熔断器
    pub fn new(failure_threshold: u32, open_duration: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                half_open_attempts: 0,
            })),
            config: CircuitBreakerConfig {
                failure_threshold,
                open_duration_secs: open_duration.as_secs(),
                ..Default::default()
            },
            open_time: Arc::new(RwLock::new(None)),
        }
    }

    /// 使用配置创建熔断器
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                half_open_attempts: 0,
            })),
            open_time: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// 获取当前状态
    pub fn state(&self) -> CircuitState {
        let mut inner = self.inner.write().unwrap();
        
        // 检查是否应该从 Open 状态转换到 Half-Open
        if inner.state == CircuitState::Open {
            if let Some(last_failure) = inner.last_failure_time {
                let elapsed = last_failure.elapsed().as_secs();
                if elapsed >= self.config.open_duration_secs {
                    inner.state = CircuitState::HalfOpen;
                    inner.half_open_attempts = 0;
                    return CircuitState::HalfOpen;
                }
            }
        }
        
        inner.state
    }

    /// 记录成功
    pub fn record_success(&self) {
        let mut inner = self.inner.write().unwrap();
        
        match inner.state {
            CircuitState::Closed => {
                // 成功时重置失败计数
                inner.failure_count = 0;
                inner.success_count += 1;
            }
            CircuitState::HalfOpen => {
                // 半开状态下成功，恢复到 Closed
                inner.success_count += 1;
                if inner.success_count >= self.config.half_open_max_attempts {
                    inner.state = CircuitState::Closed;
                    inner.failure_count = 0;
                    inner.success_count = 0;
                    inner.half_open_attempts = 0;
                }
            }
            CircuitState::Open => {
                // Open 状态下不应该有成功记录（请求被拒绝）
            }
        }
    }

    /// 记录失败
    pub fn record_failure(&self) {
        let mut inner = self.inner.write().unwrap();
        
        inner.failure_count += 1;
        inner.last_failure_time = Some(Instant::now());
        
        match inner.state {
            CircuitState::Closed => {
                // 失败次数达到阈值，打开熔断器
                if inner.failure_count >= self.config.failure_threshold {
                    inner.state = CircuitState::Open;
                    *self.open_time.write().unwrap() = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                // 半开状态下失败，重新打开熔断器
                inner.state = CircuitState::Open;
                inner.half_open_attempts = 0;
                inner.success_count = 0;
                *self.open_time.write().unwrap() = Some(Instant::now());
            }
            CircuitState::Open => {
                // Open 状态下不应该有失败记录（请求被拒绝）
            }
        }
    }

    /// 检查是否允许请求通过
    pub fn is_closed(&self) -> bool {
        self.state() == CircuitState::Closed
    }

    /// 检查熔断器是否打开
    pub fn is_open(&self) -> bool {
        let state = self.state();
        
        // 如果是 Half-Open，允许一个请求通过
        if state == CircuitState::HalfOpen {
            let mut inner = self.inner.write().unwrap();
            if inner.half_open_attempts < self.config.half_open_max_attempts {
                inner.half_open_attempts += 1;
                return false; // 允许一个请求尝试
            }
        }
        
        state == CircuitState::Open
    }

    /// 重置熔断器
    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.state = CircuitState::Closed;
        inner.failure_count = 0;
        inner.success_count = 0;
        inner.last_failure_time = None;
        inner.half_open_attempts = 0;
        *self.open_time.write().unwrap() = None;
    }

    /// 获取失败次数
    pub fn failure_count(&self) -> u32 {
        self.inner.read().unwrap().failure_count
    }

    /// 获取熔断器状态信息
    pub fn status(&self) -> CircuitBreakerStatus {
        let inner = self.inner.read().unwrap();
        let open_time = self.open_time.read().unwrap();
        
        CircuitBreakerStatus {
            state: inner.state,
            failure_count: inner.failure_count,
            success_count: inner.success_count,
            last_failure_time: inner.last_failure_time,
            open_time: *open_time,
        }
    }
}

/// 熔断器状态信息
#[derive(Debug, Clone)]
pub struct CircuitBreakerStatus {
    /// 当前状态
    pub state: CircuitState,
    /// 失败次数
    pub failure_count: u32,
    /// 成功次数
    pub success_count: u32,
    /// 最后失败时间
    pub last_failure_time: Option<Instant>,
    /// 熔断开始时间
    pub open_time: Option<Instant>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_basic() {
        let breaker = CircuitBreaker::new(3, Duration::from_secs(1));
        
        // 初始状态为 Closed
        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.is_closed());
        
        // 记录 3 次失败
        breaker.record_failure();
        breaker.record_failure();
        breaker.record_failure();
        
        // 状态变为 Open
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(breaker.is_open());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        // 使用 1 秒的开放时间，避免立即变为 HalfOpen
        let breaker = CircuitBreaker::new(2, Duration::from_secs(1));
        
        // 触发熔断
        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        
        // 等待恢复时间（留一些余量）
        std::thread::sleep(Duration::from_millis(1100));
        
        // state() 方法在超时后返回 HalfOpen
        assert_eq!(breaker.state(), CircuitState::HalfOpen);
        
        // 记录成功，恢复到 Closed
        breaker.record_success();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }
}
