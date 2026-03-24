import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { buttonTapVariants } from '@/lib/animations'

const buttonVariants = cva(
  'group relative inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all duration-300 ' +
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 ' +
    'disabled:pointer-events-none disabled:opacity-50 ' +
    'shadow-md hover:shadow-lg',
  {
    variants: {
      variant: {
        default:
          'bg-gradient-to-br from-primary to-primary/90 text-primary-foreground ' +
          'hover:from-primary/90 hover:to-primary ' +
          'shadow-primary/25 hover:shadow-primary/40',
        destructive:
          'bg-gradient-to-br from-destructive to-destructive/90 text-destructive-foreground ' +
          'hover:from-destructive/90 hover:to-destructive ' +
          'shadow-destructive/25 hover:shadow-destructive/40',
        outline:
          'border border-input/60 bg-background/60 backdrop-blur-sm ' +
          'shadow-sm hover:bg-accent/60 hover:text-accent-foreground ' +
          'hover:border-primary/30 transition-all duration-300',
        secondary:
          'bg-gradient-to-br from-secondary to-secondary/80 text-secondary-foreground ' +
          'hover:from-secondary/80 hover:to-secondary ' +
          'shadow-sm hover:shadow-md',
        ghost:
          'bg-transparent hover:bg-accent/60 hover:text-accent-foreground ' +
          'shadow-none hover:shadow-sm',
        link: 'text-primary underline-offset-4 hover:underline',
        acrylic:
          'bg-white/70 dark:bg-gray-900/60 backdrop-blur-xl ' +
          'border border-white/20 dark:border-gray-700/30 ' +
          'text-foreground shadow-md hover:shadow-lg ' +
          'hover:bg-white/80 dark:hover:bg-gray-900/70 ' +
          'transition-all duration-300',
        glass:
          'bg-white/10 backdrop-blur-lg ' +
          'border border-white/20 ' +
          'text-foreground shadow-inner ' +
          'hover:bg-white/20 hover:shadow-md ' +
          'transition-all duration-300',
      },
      size: {
        default: 'h-10 px-5 py-2.5',
        sm: 'h-9 rounded-md px-4 text-xs',
        lg: 'h-12 rounded-md px-8 text-base',
        icon: 'h-10 w-10',
        xl: 'h-14 rounded-md px-10 text-lg',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
  loading?: boolean
  leftIcon?: React.ReactNode
  rightIcon?: React.ReactNode
  shine?: boolean
  glow?: boolean
  animate?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      asChild = false,
      loading,
      leftIcon,
      rightIcon,
      children,
      shine = false,
      glow = false,
      animate = true,
      ...props
    },
    ref
  ) => {
    const Comp = asChild ? Slot : 'button'
    
    const buttonContent = (
      <>
        {/* 光晕层 */}
        {variant === 'default' && (
          <motion.div
            className="absolute inset-0 rounded-md bg-gradient-to-r from-white/0 via-white/10 to-white/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
            initial={{ x: '-100%' }}
            whileHover={{ x: '100%' }}
            transition={{ duration: 0.5 }}
          />
        )}

        {loading && (
          <motion.svg
            className="mr-2 h-4 w-4 animate-spin"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </motion.svg>
        )}
        {!loading && leftIcon && (
          <motion.span
            className="transition-transform duration-300"
            whileHover={{ scale: 1.1, x: -2 }}
          >
            {leftIcon}
          </motion.span>
        )}
        {children}
        {!loading && rightIcon && (
          <motion.span
            className="transition-transform duration-300"
            whileHover={{ scale: 1.1, x: 2 }}
          >
            {rightIcon}
          </motion.span>
        )}
      </>
    )

    if (animate && !loading) {
      return (
        <Comp
          className={cn(
            buttonVariants({ variant, size, className }),
            shine && 'btn-shine',
            glow && 'hover-glow'
          )}
          ref={ref}
          disabled={props.disabled}
          {...props}
        >
          <motion.div
            className="w-full h-full flex items-center justify-center"
            variants={buttonTapVariants}
            whileTap="tap"
            initial="rest"
            animate="rest"
          >
            {buttonContent}
          </motion.div>
        </Comp>
      )
    }

    return (
      <Comp
        className={cn(
          buttonVariants({ variant, size, className }),
          shine && 'btn-shine',
          glow && 'hover-glow'
        )}
        ref={ref}
        disabled={loading || props.disabled}
        {...props}
      >
        {buttonContent}
      </Comp>
    )
  }
)
Button.displayName = 'Button'

// Slot 组件用于 asChild 功能
const Slot = React.forwardRef<
  HTMLSpanElement,
  React.HTMLAttributes<HTMLSpanElement>
>(({ children, ...props }, ref) => {
  return (
    <span ref={ref} {...props}>
      {children}
    </span>
  )
})
Slot.displayName = 'Slot'

export { Button, buttonVariants }
