import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value: number, decimals: number = 2): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)
}

export function formatPercent(value: number, decimals: number = 2): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`
}

export function formatNumber(value: number, decimals: number = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value)
}

export function formatDate(date: string | Date | null | undefined): string {
  if (!date) return 'N/A'

  const parsedDate = new Date(date)

  // Check if the date is valid
  if (isNaN(parsedDate.getTime())) {
    return 'Invalid date'
  }

  return new Intl.DateTimeFormat('en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsedDate)
}

export function getSignalColor(signal: string): string {
  switch (signal?.toUpperCase()) {
    case 'BUY':
    case 'BULLISH':
      return 'text-green-500'
    case 'SELL':
    case 'BEARISH':
      return 'text-red-500'
    default:
      return 'text-yellow-500'
  }
}

export function getProfitColor(profit: number): string {
  if (profit > 0) return 'text-green-500'
  if (profit < 0) return 'text-red-500'
  return 'text-muted-foreground'
}
