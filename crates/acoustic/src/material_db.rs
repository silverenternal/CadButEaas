//! 材料吸声系数数据库
//!
//! 从配置文件加载材料吸声系数，支持：
//! - TOML 格式配置文件
//! - 频率相关的吸声系数
//! - 材料密度、厚度等可选参数

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tracing::{info, warn};

use crate::acoustic_types::Frequency;

/// 材料吸声系数数据库
#[derive(Debug, Clone)]
pub struct MaterialDatabase {
    materials: BTreeMap<String, MaterialProps>,
}

/// 材料属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProps {
    /// 材料名称
    pub name: String,
    /// 吸声系数（频率相关）
    pub absorption_coeffs: BTreeMap<Frequency, f64>,
    /// 密度 (kg/m³)，可选
    #[serde(default)]
    pub density: Option<f64>,
    /// 厚度 (mm)，可选
    #[serde(default)]
    pub thickness: Option<f64>,
    /// 描述，可选
    #[serde(default)]
    pub description: Option<String>,
}

/// TOML 配置文件格式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialDatabaseToml {
    /// 材料列表
    pub materials: Vec<MaterialPropsToml>,
}

/// TOML 格式的材料属性（使用字符串键）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialPropsToml {
    /// 材料名称
    pub name: String,
    /// 吸声系数（使用字符串键，如 "125_hz"）
    pub absorption_coeffs: BTreeMap<String, f64>,
    /// 密度 (kg/m³)，可选
    #[serde(default)]
    pub density: Option<f64>,
    /// 厚度 (mm)，可选
    #[serde(default)]
    pub thickness: Option<f64>,
    /// 描述，可选
    #[serde(default)]
    pub description: Option<String>,
}

impl MaterialPropsToml {
    /// 转换为 MaterialProps
    fn into_props(self) -> Result<MaterialProps, String> {
        let mut coeffs = BTreeMap::new();

        for (key, value) in self.absorption_coeffs {
            let freq = match key.to_lowercase().as_str() {
                "125_hz" | "hz125" | "125" => Frequency::Hz125,
                "250_hz" | "hz250" | "250" => Frequency::Hz250,
                "500_hz" | "hz500" | "500" => Frequency::Hz500,
                "1k_hz" | "hz1k" | "1000" | "1k" => Frequency::Hz1k,
                "2k_hz" | "hz2k" | "2000" | "2k" => Frequency::Hz2k,
                "4k_hz" | "hz4k" | "4000" | "4k" => Frequency::Hz4k,
                _ => return Err(format!("未知频率：{}", key)),
            };
            coeffs.insert(freq, value);
        }

        Ok(MaterialProps {
            name: self.name,
            absorption_coeffs: coeffs,
            density: self.density,
            thickness: self.thickness,
            description: self.description,
        })
    }
}

impl MaterialDatabase {
    /// 创建新的 MaterialDatabase（空）
    pub fn new() -> Self {
        Self {
            materials: BTreeMap::new(),
        }
    }

    /// 从 TOML 文件加载材料数据库
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        info!("从文件加载材料数据库：{:?}", path);

        let content = fs::read_to_string(path).map_err(|e| format!("读取文件失败：{}", e))?;

        Self::load_from_toml(&content)
    }

    /// 从 TOML 字符串加载材料数据库
    pub fn load_from_toml(content: &str) -> Result<Self, String> {
        let toml_db: MaterialDatabaseToml =
            toml::from_str(content).map_err(|e| format!("解析 TOML 失败：{}", e))?;

        let mut materials = BTreeMap::new();

        for material_toml in toml_db.materials {
            let props = material_toml.into_props()?;
            // 存储多个键名以便查找
            let name_lower = props.name.to_lowercase();
            materials.insert(name_lower.clone(), props.clone());

            // 如果有中文名，也添加中文键
            if props.name.contains('(') {
                if let Some(cn_name) = props
                    .name
                    .split('(')
                    .nth(1)
                    .and_then(|s| s.split(')').next())
                {
                    materials.insert(cn_name.to_lowercase(), props);
                }
            }
        }

        info!("加载了 {} 种材料", materials.len());

        Ok(Self { materials })
    }

    /// 获取材料的吸声系数
    pub fn get_absorption_coeffs(&self, material_name: &str) -> Option<&BTreeMap<Frequency, f64>> {
        let key = material_name.to_lowercase();
        self.materials.get(&key).map(|m| &m.absorption_coeffs)
    }

    /// 获取材料属性
    pub fn get_material(&self, material_name: &str) -> Option<&MaterialProps> {
        let key = material_name.to_lowercase();
        self.materials.get(&key)
    }

    /// 检查材料是否存在
    pub fn contains(&self, material_name: &str) -> bool {
        let key = material_name.to_lowercase();
        self.materials.contains_key(&key)
    }

    /// 获取所有材料名称
    pub fn material_names(&self) -> Vec<&str> {
        self.materials.keys().map(|k| k.as_str()).collect()
    }

    /// 添加材料
    pub fn add_material(&mut self, props: MaterialProps) {
        let key = props.name.to_lowercase();
        self.materials.insert(key, props);
    }

    /// 使用内置默认材料创建数据库
    pub fn with_defaults() -> Self {
        let toml_content = include_str!("../materials.default.toml");
        Self::load_from_toml(toml_content).unwrap_or_else(|e| {
            warn!("加载默认材料失败：{}", e);
            Self::new()
        })
    }
}

impl Default for MaterialDatabase {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_database_creation() {
        let db = MaterialDatabase::new();
        assert!(db.materials.is_empty());
    }

    #[test]
    fn test_material_database_from_toml() {
        let toml_content = r#"
[[materials]]
name = "concrete"
density = 2400.0
description = "混凝土"

[materials.absorption_coeffs]
125_hz = 0.01
250_hz = 0.01
500_hz = 0.02
1k_hz = 0.02
2k_hz = 0.02
4k_hz = 0.03
"#;

        let db = MaterialDatabase::load_from_toml(toml_content).unwrap();
        assert!(db.contains("concrete"));

        let coeffs = db.get_absorption_coeffs("concrete").unwrap();
        assert!((coeffs.get(&Frequency::Hz125).unwrap() - 0.01).abs() < 1e-10);
        assert!((coeffs.get(&Frequency::Hz500).unwrap() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_material_database_default() {
        let db = MaterialDatabase::default();
        // 默认数据库应该包含一些常见材料
        assert!(db.contains("concrete") || db.contains("混凝土"));
    }

    #[test]
    fn test_material_props_conversion() {
        let toml_props = MaterialPropsToml {
            name: "test".to_string(),
            absorption_coeffs: [
                ("125".to_string(), 0.1),
                ("250".to_string(), 0.2),
                ("500".to_string(), 0.3),
                ("1000".to_string(), 0.4),
                ("2000".to_string(), 0.5),
                ("4000".to_string(), 0.6),
            ]
            .into_iter()
            .collect(),
            density: Some(1000.0),
            thickness: Some(10.0),
            description: Some("测试材料".to_string()),
        };

        let props = toml_props.into_props().unwrap();
        assert_eq!(props.name, "test");
        assert_eq!(props.density, Some(1000.0));
        assert!(props.absorption_coeffs.contains_key(&Frequency::Hz125));
    }

    #[test]
    fn test_invalid_frequency() {
        let toml_props = MaterialPropsToml {
            name: "test".to_string(),
            absorption_coeffs: [("invalid".to_string(), 0.1)].into_iter().collect(),
            density: None,
            thickness: None,
            description: None,
        };

        let result = toml_props.into_props();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("未知频率"));
    }
}
