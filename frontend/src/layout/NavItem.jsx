/**
 * src/layout/NavItem.jsx
 * ----------------------
 * One sidebar navigation entry, built on react-router's NavLink so the active
 * route is highlighted automatically (.nav-item-active is defined in index.css).
 */

import React from 'react'
import { NavLink } from 'react-router-dom'

export default function NavItem({ to, icon, children, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) => ['nav-item', isActive ? 'nav-item-active' : ''].join(' ')}
    >
      {icon}
      <span>{children}</span>
    </NavLink>
  )
}
