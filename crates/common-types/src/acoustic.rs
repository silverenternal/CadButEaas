//! 声学分析共享类型（仅 Frequency）
//!
//! 本模块仅保留 `Frequency` 枚举，因为 `Material` 类型需要它。
//! 其他声学类型已迁移至 `acoustic` crate，请使用 `acoustic::` 路径。

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// 频率（倍频程）
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Frequency {
    /// 125 Hz
    Hz125,
    /// 250 Hz
    Hz250,
    /// 500 Hz
    Hz500,
    /// 1000 Hz
    Hz1k,
    /// 2000 Hz
    Hz2k,
    /// 4000 Hz
    Hz4k,
}

impl Frequency {
    /// 转换为 Hz
    pub fn to_hz(self) -> f64 {
        match self {
            Frequency::Hz125 => 125.0,
            Frequency::Hz250 => 250.0,
            Frequency::Hz500 => 500.0,
            Frequency::Hz1k => 1000.0,
            Frequency::Hz2k => 2000.0,
            Frequency::Hz4k => 4000.0,
        }
    }

    /// 从 Hz 值创建 Frequency（就近匹配）
    pub fn from_hz(hz: f64) -> Self {
        let frequencies = [
            (125.0, Frequency::Hz125),
            (250.0, Frequency::Hz250),
            (500.0, Frequency::Hz500),
            (1000.0, Frequency::Hz1k),
            (2000.0, Frequency::Hz2k),
            (4000.0, Frequency::Hz4k),
        ];

        frequencies
            .iter()
            .min_by(|(f1, _), (f2, _)| (f1 - hz).abs().partial_cmp(&(f2 - hz).abs()).unwrap())
            .map(|(_, freq)| *freq)
            .unwrap_or(Frequency::Hz500)
    }

    /// 获取所有频率（按顺序）
    pub fn all() -> Vec<Self> {
        vec![
            Frequency::Hz125,
            Frequency::Hz250,
            Frequency::Hz500,
            Frequency::Hz1k,
            Frequency::Hz2k,
            Frequency::Hz4k,
        ]
    }
}

impl std::fmt::Display for Frequency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.0} Hz", self.to_hz())
    }
}
