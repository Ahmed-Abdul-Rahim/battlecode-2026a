(module
  (import "env" "fuel_pause" (func $fuel_pause (result i32)))
  (import "env" "fuel_resume" (func $fuel_resume))
  (func (export "cambc_debug_enabled") (param $_ (ref eq)) (result (ref eq))
    (ref.i31 (i32.const 1)))
  (func (export "cambc_fuel_pause") (param $_ (ref eq)) (result (ref eq))
    (ref.i31 (call $fuel_pause)))
  (func (export "cambc_fuel_resume") (param $_ (ref eq)) (result (ref eq))
    (call $fuel_resume) (ref.i31 (i32.const 0))))
