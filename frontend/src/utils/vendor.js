export function vendorName(vendor) {
  if (!vendor) return 'Unknown';
  if (typeof vendor === 'string') return vendor;
  if (typeof vendor === 'object') {
    if (vendor.name) return vendor.name;
    // fallback: try common fields
    if (vendor.vendor_name) return vendor.vendor_name;
    if (vendor.displayName) return vendor.displayName;
    try {
      return JSON.stringify(vendor);
    } catch (e) {
      return 'Unknown';
    }
  }
  return String(vendor);
}
