# -*- coding: utf-8 -*-
"""
/***************************************************************************
Name                 : MapSwipe tool
Description          : Plugin for swipe active layer
Date                 : October, 2015
copyright            : (C) 2015 by Hirofumi Hayashi and Luiz Motta
email                : hayashi@apptec.co.jp and motta.luiz@gmail.com

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

from qgis.PyQt.QtCore import Qt, QPoint
from qgis.PyQt.QtGui import QCursor

from qgis.gui import QgsMapTool

from .swipemap import SwipeMap

class MapSwipeTool(QgsMapTool):
  def __init__(self, canvas):
    self.canvas = canvas
    super().__init__( canvas )
    self.swipe = SwipeMap( self.canvas )
    self.checkDirection =  self.hasSwipe = self.disabledSwipe = None
    self.firstPoint = QPoint()
    self.cursorV = QCursor( Qt.SplitVCursor )
    self.cursorH = QCursor( Qt.SplitHCursor )
 

  def canvasPressEvent(self, e):
      self.hasSwipe = True
      self.firstPoint.setX( e.x() )
      self.firstPoint.setY( e.y() )
      self.checkDirection = True
      self.canvas.setCursor(QCursor(Qt.PointingHandCursor))
      #self.hasSwipe = False
      #self.disabledSwipe = False
      self.swipe.clear()
      self.swipe.setLayers()
      self.swipe.setMap()

  def canvasReleaseEvent(self, e):
    self.hasSwipe = False
    self.canvas.setCursor(QCursor(Qt.PointingHandCursor))
    self.deactivated.emit()
    self.swipe.clear()
    
  def canvasMoveEvent(self, e):
    if self.hasSwipe:
      if self.checkDirection:
        dX = abs( e.x() - self.firstPoint.x() )
        dY = abs( e.y() - self.firstPoint.y() )
        isVertical = dX > dY
        self.swipe.setIsVertical( isVertical )
        self.checkDirection = False
        self.canvas.setCursor( self.cursorH if isVertical else self.cursorV )
        
    self.swipe.setLength( e.x(), e.y() )

  def disable(self):
    self.swipe.clear()
    self.hasSwipe = False
    self.disabledSwipe = True
