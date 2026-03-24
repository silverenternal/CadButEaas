import type { Meta, StoryObj } from '@storybook/react'
import { Button } from './button'
import { FolderOpen } from 'lucide-react'

const meta = {
  title: 'UI/Button',
  component: Button,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'destructive', 'outline', 'secondary', 'ghost', 'link'],
    },
    size: {
      control: 'select',
      options: ['default', 'sm', 'lg', 'icon'],
    },
  },
  args: { onClick: undefined },
} satisfies Meta<typeof Button>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    children: 'Button',
    variant: 'default',
  },
}

export const Destructive: Story = {
  args: {
    children: 'Delete',
    variant: 'destructive',
  },
}

export const Outline: Story = {
  args: {
    children: 'Cancel',
    variant: 'outline',
  },
}

export const Ghost: Story = {
  args: {
    children: 'Maybe',
    variant: 'ghost',
  },
}

export const WithIcon: Story = {
  args: {
    children: 'Open File',
    leftIcon: <FolderOpen className="h-4 w-4" />,
  },
}

export const Loading: Story = {
  args: {
    children: 'Loading',
    loading: true,
  },
}

export const IconOnly: Story = {
  args: {
    children: <FolderOpen className="h-4 w-4" />,
    size: 'icon',
    variant: 'ghost',
  },
}

export const Small: Story = {
  args: {
    children: 'Small Button',
    size: 'sm',
  },
}

export const Large: Story = {
  args: {
    children: 'Large Button',
    size: 'lg',
  },
}
