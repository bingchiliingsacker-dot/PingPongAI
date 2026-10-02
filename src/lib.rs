mod environment;

use pyo3::prelude::*;

#[pymodule]
fn ping_pong_env_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Environment is now an internal Rust struct; only Simulation is exposed.
    m.add_class::<environment::Simulation>()?;
    Ok(())
}
