"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface SliderProps {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  className?: string
  showValue?: boolean
  formatValue?: (value: number) => string
}

export function Slider({
  value,
  onValueChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  className,
  showValue = true,
  formatValue = (v) => `${v}%`,
}: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="relative flex-1 h-2">
        <div className="absolute inset-0 rounded-full bg-muted" />
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-primary"
          style={{ width: `${percentage}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onValueChange(Number(e.target.value))}
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-primary border-2 border-background shadow-sm pointer-events-none"
          style={{ left: `calc(${percentage}% - 8px)` }}
        />
      </div>
      {showValue && (
        <span className="text-sm font-medium text-muted-foreground min-w-[3rem] text-right">
          {formatValue(value)}
        </span>
      )}
    </div>
  )
}
