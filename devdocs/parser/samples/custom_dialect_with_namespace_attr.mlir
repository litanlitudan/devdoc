module {
  func.func @main() {
    // Custom op carries a dialect-specific namespace attribute.
    // A handler can read `your.namespace` to set the node's hierarchical name.
    %0 = "mydialect.foo"() { your.namespace = "alpha/beta/gamma" } : () -> ()
    return
  }
}

