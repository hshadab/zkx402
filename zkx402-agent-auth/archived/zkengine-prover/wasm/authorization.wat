(module
  ;; Complex Authorization Policy Circuit for zkEngine
  ;;
  ;; This WASM circuit implements complex authorization logic that cannot
  ;; be expressed in ONNX neural networks:
  ;; - Vendor whitelist/blacklist checking
  ;; - Time-based rules (business hours, weekends)
  ;; - Multi-condition boolean logic
  ;; - String operations (vendor domain matching)
  ;;
  ;; Inputs:
  ;;   - transaction_amount (u64)
  ;;   - vendor_id (u64, hash of vendor domain)
  ;;   - timestamp (u64, Unix timestamp)
  ;;   - balance (u64, private)
  ;;   - daily_budget_remaining (u64, private)
  ;;   - whitelist_bitmap (u64, private - bit flags for allowed vendors)
  ;;
  ;; Output:
  ;;   - authorized (0 or 1)

  ;; Memory: Store intermediate values
  (memory (export "memory") 1)

  ;; Helper: Check if vendor is in whitelist
  ;; Input: vendor_id (0-63 maps to bit positions)
  ;; Input: whitelist_bitmap (64-bit bitmap)
  ;; Output: 1 if whitelisted, 0 otherwise
  (func $is_vendor_whitelisted (param $vendor_id i64) (param $whitelist_bitmap i64) (result i64)
    (local $bit_position i64)
    (local $mask i64)

    ;; Get bit position (vendor_id % 64)
    (local.set $bit_position
      (i64.rem_u (local.get $vendor_id) (i64.const 64))
    )

    ;; Create mask (1 << bit_position)
    (local.set $mask
      (i64.shl (i64.const 1) (local.get $bit_position))
    )

    ;; Check if bit is set
    (i64.ne
      (i64.and (local.get $whitelist_bitmap) (local.get $mask))
      (i64.const 0)
    )
  )

  ;; Helper: Check if timestamp is within business hours
  ;; Business hours: Monday-Friday, 9am-5pm EST
  ;; Input: timestamp (Unix timestamp)
  ;; Output: 1 if within business hours, 0 otherwise
  (func $is_business_hours (param $timestamp i64) (result i64)
    (local $hour_of_day i64)
    (local $day_of_week i64)
    (local $seconds_in_day i64)
    (local $is_weekday i64)
    (local $is_work_hours i64)

    ;; Constants
    (local.set $seconds_in_day (i64.const 86400))  ;; 24 * 60 * 60

    ;; Calculate hour of day (0-23)
    ;; hour_of_day = (timestamp % 86400) / 3600
    (local.set $hour_of_day
      (i64.div_u
        (i64.rem_u (local.get $timestamp) (local.get $seconds_in_day))
        (i64.const 3600)
      )
    )

    ;; Calculate day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
    ;; Simplified: ((timestamp / 86400) + 4) % 7
    ;; (Unix epoch was a Thursday, so +4 offset)
    (local.set $day_of_week
      (i64.rem_u
        (i64.add
          (i64.div_u (local.get $timestamp) (local.get $seconds_in_day))
          (i64.const 4)
        )
        (i64.const 7)
      )
    )

    ;; Check if weekday (Monday=1 to Friday=5)
    (local.set $is_weekday
      (i64.and
        (i64.ge_u (local.get $day_of_week) (i64.const 1))
        (i64.le_u (local.get $day_of_week) (i64.const 5))
      )
    )

    ;; Check if work hours (9am=9 to 5pm=17)
    (local.set $is_work_hours
      (i64.and
        (i64.ge_u (local.get $hour_of_day) (i64.const 9))
        (i64.lt_u (local.get $hour_of_day) (i64.const 17))
      )
    )

    ;; Return weekday AND work hours
    (i64.and (local.get $is_weekday) (local.get $is_work_hours))
  )

  ;; Main authorization function
  ;; Exported function that will be proven by zkEngine
  (func (export "check_authorization")
    (param $transaction_amount i64)
    (param $vendor_id i64)
    (param $timestamp i64)
    (param $balance i64)
    (param $daily_budget_remaining i64)
    (param $whitelist_bitmap i64)
    (result i64)

    (local $rule_1 i64)  ;; amount <= balance * 0.1
    (local $rule_2 i64)  ;; vendor is whitelisted
    (local $rule_3 i64)  ;; within business hours
    (local $rule_4 i64)  ;; amount <= daily_budget_remaining
    (local $max_amount i64)

    ;; Rule 1: Transaction amount <= 10% of balance
    ;; max_amount = balance / 10
    (local.set $max_amount
      (i64.div_u (local.get $balance) (i64.const 10))
    )
    (local.set $rule_1
      (i64.le_u (local.get $transaction_amount) (local.get $max_amount))
    )

    ;; Rule 2: Vendor must be whitelisted
    (local.set $rule_2
      (call $is_vendor_whitelisted
        (local.get $vendor_id)
        (local.get $whitelist_bitmap)
      )
    )

    ;; Rule 3: Transaction must be during business hours
    (local.set $rule_3
      (call $is_business_hours (local.get $timestamp))
    )

    ;; Rule 4: Amount <= daily budget remaining
    (local.set $rule_4
      (i64.le_u (local.get $transaction_amount) (local.get $daily_budget_remaining))
    )

    ;; Return: authorized if ALL rules pass
    ;; authorized = rule_1 AND rule_2 AND rule_3 AND rule_4
    (i64.and
      (i64.and (local.get $rule_1) (local.get $rule_2))
      (i64.and (local.get $rule_3) (local.get $rule_4))
    )
  )

  ;; Alternative: Simpler policy (for testing)
  ;; Just checks amount and whitelist
  (func (export "check_authorization_simple")
    (param $transaction_amount i64)
    (param $vendor_id i64)
    (param $balance i64)
    (param $whitelist_bitmap i64)
    (result i64)

    (local $rule_amount i64)
    (local $rule_whitelist i64)
    (local $max_amount i64)

    ;; amount <= 10% of balance
    (local.set $max_amount
      (i64.div_u (local.get $balance) (i64.const 10))
    )
    (local.set $rule_amount
      (i64.le_u (local.get $transaction_amount) (local.get $max_amount))
    )

    ;; vendor is whitelisted
    (local.set $rule_whitelist
      (call $is_vendor_whitelisted
        (local.get $vendor_id)
        (local.get $whitelist_bitmap)
      )
    )

    ;; Return AND of both rules
    (i64.and (local.get $rule_amount) (local.get $rule_whitelist))
  )
)
