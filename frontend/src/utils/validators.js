export const validateFile = (file) => {
  const allowedTypes = [
    'application/pdf',
    'image/png',
    'image/jpeg',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel'
  ];

  const maxSize = 10 * 1024 * 1024; // 10MB

  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      error: 'File type not supported. Please upload PDF, PNG, JPG, or Excel files.'
    };
  }

  if (file.size > maxSize) {
    return {
      valid: false,
      error: 'File size too large. Maximum size is 10MB.'
    };
  }

  return { valid: true };
};

export const validateInvoiceAmount = (amount) => {
  const numAmount = parseFloat(amount);
  
  if (isNaN(numAmount)) {
    return {
      valid: false,
      error: 'Amount must be a valid number'
    };
  }

  if (numAmount <= 0) {
    return {
      valid: false,
      error: 'Amount must be greater than zero'
    };
  }

  if (numAmount > 1000000) {
    return {
      valid: false,
      error: 'Amount exceeds maximum limit of $1,000,000'
    };
  }

  return { valid: true };
};

export const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  
  if (!emailRegex.test(email)) {
    return {
      valid: false,
      error: 'Please enter a valid email address'
    };
  }

  return { valid: true };
};