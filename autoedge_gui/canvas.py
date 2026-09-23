"""可缩放/平移的图像显示控件。

支持：
* 滚轮以光标为中心缩放，中键或空格拖动平移；
* 以"图像坐标"上报鼠标位置，便于状态栏显示与编辑窗口的命中测试；
* 显示叠加参考线（用于编辑窗口的选中高亮）。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

__all__ = ["ImageCanvas", "bgr_to_qimage"]


def bgr_to_qimage(bgr: np.ndarray, alpha: np.ndarray | None = None) -> QImage:
    """把 BGR（可选带 alpha）的 numpy 数组转成 ``QImage``。"""
    arr = np.ascontiguousarray(bgr)
    h, w = arr.shape[:2]
    if alpha is not None:
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = arr[:, :, ::-1]
        rgba[:, :, 3] = alpha
        return QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    rgb = np.ascontiguousarray(arr[:, :, ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageCanvas(QGraphicsView):
    """可缩放平移的图像画布，并支持 ROI 圈框与提示点标注。

    交互模式：
    * ``"pan"``    —— 浏览（滚轮缩放，Shift+左键或中键拖动平移）
    * ``"roi"``    —— 左键逐点画多边形，右键/Esc 结束当前多边形，双击结束
    * ``"hint+"``  —— 左键放置绿点（"这里是晶体"）
    * ``"hint-"``  —— 左键放置红点（"这里不是晶体"）
    """

    #: 鼠标在**原图坐标系**中的位置
    cursor_moved = Signal(float, float)
    #: 图像被点击（左键），参数为图像坐标
    clicked = Signal(float, float)
    #: 左键在画布上按下并移动（拖动），参数为**图像坐标**
    dragged = Signal(float, float)
    #: 鼠标左键松开（结束拖动）
    drag_released = Signal()
    #: 鼠标右键按下，参数为**图像坐标**（编辑窗口用它结束一条线）
    right_clicked = Signal(float, float)
    #: 视图缩放倍率变化
    zoom_changed = Signal(float)
    #: 一个多边形完成（参数为顶点列表，**原图坐标**）
    polygon_committed = Signal(list)
    #: 正处于绘制中的多边形顶点变化（用于实时预览）
    polygon_in_progress = Signal(list)
    #: 在提示点模式下右键，请求删除离 (x, y) 最近的提示点（**原图坐标**）
    hint_remove_requested = Signal(float, float)

    #: 视图允许的最小/最大缩放倍率（相对像素 1:1）。
    #: 上限 64 是"看细节"用的：4928px 的原图切到 1/4 预览（1600px）后，
    #: 64 倍相当于盯着约 6 个原图像素看，足够看清边界。
    MIN_ZOOM = 0.02
    MAX_ZOOM = 64.0

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.darkGray)
        self.setMouseTracking(True)

        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.SmoothTransformation)
        self._scene.addItem(self._item)
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._pixmap = QPixmap()
        self._panning = False
        self._pan_start = QPoint()
        self._fit_on_next = True
        self._image_size = (0, 0)
        #: 当前贴上去的图覆盖原图的哪一块 ``(x0, y0, x1, y1)``（整幅预览时就是整张图）
        self._image_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        #: 原图尺寸
        self._original_size: tuple[int, int] | None = None
        #: 当前显示图在场景中的左上角、尺寸，以及它对应的原图左上角像素
        self._scene_origin: tuple[float, float] = (0.0, 0.0)
        self._scene_span: tuple[float, float] = (0.0, 0.0)
        #: 使用者是否已经自己缩放/平移过（之后 resize 不再自动适应）
        self._user_viewed = False
        #: 当前交互模式
        self.mode = "pan"
        #: 正在绘制中的多边形顶点（图像坐标）
        self._pending: list[tuple[float, float]] = []
        #: 最近一次框选的原始数据（用于核对坐标换算）
        self.last_polygon_debug: dict = {}
        #: 最近一次单点点击的原始数据
        self._last_click_debug: dict = {}
        #: 本次框选每个顶点的三级坐标
        self._pending_clicks: list[dict] = []
        self._first_point_scene: tuple[float, float] | None = None
        self._first_point_viewport: tuple[int, int] | None = None

    # ------------------------------------------------------------------ 坐标
    def to_image(self, x: float, y: float) -> tuple[float, float]:
        """把场景坐标换算成**原图坐标**。

        映射只依赖两件事（这样最不容易搞错，也不需要区分"整幅/切片"）：

        * 当前贴上去的图覆盖原图的哪一块 —— ``_image_rect = (x0, y0, x1, y1)``
        * 它在场景里占多大 —— ``_scene_span``，位置在 ``_scene_origin``

        于是 ``比例 = 覆盖宽度 / 场景跨度``：整幅预览时覆盖宽度就是整张原图，
        放大成切片时覆盖宽度就是那一块 —— 两种情况同一个公式。
        """
        x0, y0, x1, y1 = self._image_rect
        sw, sh = self._scene_span
        if sw <= 0 or sh <= 0:
            return x, y
        return (
            x0 + (x - self._scene_origin[0]) * (x1 - x0) / sw,
            y0 + (y - self._scene_origin[1]) * (y1 - y0) / sh,
        )

    def to_scene(self, x: float, y: float) -> tuple[float, float]:
        """把**原图坐标**换算成场景坐标（用于绘制叠加层）。"""
        x0, y0, x1, y1 = self._image_rect
        sw, sh = self._scene_span
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0 or sw <= 0 or sh <= 0:
            return x, y
        return (
            self._scene_origin[0] + (x - x0) * sw / w,
            self._scene_origin[1] + (y - y0) * sh / h,
        )

    def scene_bounds_ok(self, x: float, y: float, tol: float = 1.0) -> bool:
        """判断场景坐标是否落在场景范围内（自检用）。"""
        sr = self.sceneRect()
        return (-tol <= x <= sr.width() + tol) and (-tol <= y <= sr.height() + tol)

    def image_to_display_scale(self) -> float:
        """显示图相对原图的缩放（用于线宽等按像素缩放的量，也记进诊断日志）。

        放大成切片后，"显示图/原图"没有单一含义，这里统一返回
        **整幅预览**的比例：``场景跨度 / 覆盖的原图宽度``。
        """
        x0, y0, x1, y1 = self._image_rect
        w = x1 - x0
        if w <= 0:
            return 1.0
        return float(self._scene_span[0]) / float(w)

    # ------------------------------------------------------------------ 内容
    def set_image(
        self,
        image: QImage,
        keep_view: bool = False,
        original_size: tuple[int, int] | None = None,
        keep_pending: bool = True,
    ) -> None:
        """设置显示的图像（整幅图口径）。

        Args:
            image: 要显示的图像（可能是降采样后的预览）。
            keep_view: 是否保持当前缩放/平移。
            original_size: **原图**的 ``(宽, 高)``。坐标换算由"显示图尺寸 / 原图尺寸"
                得出，因此必须提供；不提供则视为原尺寸显示。
            keep_pending: 是否保留正在绘制中的多边形。**刷新同一张图的叠加层时
                必须为 True**，否则画框画到一半会被自己清掉。
        """
        self._image_size = (image.width(), image.height())
        self._original_size = original_size or self._image_size
        self._scene_origin = (0.0, 0.0)
        self._scene_span = (float(image.width()), float(image.height()))
        self._image_rect = (0.0, 0.0, float(self._original_size[0]), float(self._original_size[1]))
        self._pixmap = QPixmap.fromImage(image)
        self._item.setPixmap(self._pixmap)
        self._item.setPos(0.0, 0.0)
        self._scene.setSceneRect(0, 0, image.width(), image.height())
        if not keep_pending:
            self._pending = []
        if not keep_view:
            self.fit_to_window()

    # ------------------------------------------------------------------ 叠加层
    def set_overlay(
        self,
        bbox: tuple[int, int, int, int] | None,
        bgra: np.ndarray | None,
        keep_view: bool = True,
    ) -> None:
        """设置一层"覆盖在图上的小图"（如擦除圆圈光标），**不动底图**。

        擦除时鼠标每动一下就要更新圆圈，如果每次都重画整张底图会非常卡；
        用独立的图元只贴一小块，代价与圆圈大小成正比。

        Args:
            bbox: 该小块在**显示图**坐标系里的 ``(x0, y0, x1, y1)``（右开）。
            bgra: ``(h, w, 4)`` uint8，BGRA 顺序（与 OpenCV 一致）。
            keep_view: 是否保持当前缩放/平移。
        """
        if bbox is None or bgra is None:
            self.clear_overlay()
            return
        x0, y0, x1, y1 = (int(v) for v in bbox)
        if x1 <= x0 or y1 <= y0:
            self.clear_overlay()
            return
        arr = np.ascontiguousarray(bgra)
        h, w = arr.shape[:2]
        qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_ARGB32).copy()
        if self._overlay_item is None:
            self._overlay_item = QGraphicsPixmapItem()
            self._overlay_item.setTransformationMode(Qt.SmoothTransformation)
            self._overlay_item.setZValue(10)
            self._scene.addItem(self._overlay_item)
        self._overlay_item.setPixmap(QPixmap.fromImage(qimg))
        self._overlay_item.setPos(x0, y0)
        self._overlay_item.setVisible(True)

    def clear_overlay(self) -> None:
        """移除叠加层。"""
        if self._overlay_item is not None:
            self._overlay_item.setVisible(False)
            self._overlay_item.setPixmap(QPixmap())

    # ------------------------------------------------------------------ 高分辨率
    def magnification(self) -> float:
        """原图 1 个像素现在在屏幕上占几个像素。

        判断"要不要按原图重新画"必须用这个值：
        单看"缓冲尺寸/原图"永远是降采样的比例，永远触发不了高分辨率。
        """
        sw = self._scene_span[0]
        if sw <= 0 or self._original_size is None:
            return 1.0
        ow = self._original_size[0]
        return self.current_scale() * (sw / ow)

    def display_scale(self) -> float:
        """整幅预览相对原图的比例（放大率计算用）。"""
        return self.image_to_display_scale()

    def rendered_image_rect(self) -> tuple[float, float, float, float] | None:
        """当前这张图覆盖原图的哪一块（``x0, y0, x1, y1``）。"""
        x0, y0, x1, y1 = self._image_rect
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return None
        return (x0, y0, x1, y1)

    def set_scene_image(
        self,
        image: QImage,
        scene_origin: tuple[float, float],
        scene_span: tuple[float, float],
        image_rect: tuple[float, float, float, float],
        scene_size: tuple[float, float],
        anchor_scene: tuple[float, float] | None = None,
        anchor_image: tuple[float, float] | None = None,
    ) -> None:
        """放置一张图，并显式声明它**覆盖原图的哪一块**。

        用途：放大后按可见区域用原图重画，于是在 1:1 甚至更大时看到的仍是真实像素，
        而不是把降采样预览摊大。坐标映射由"覆盖的原图区域 + 场景跨度"唯一确定，
        整幅预览与放大切片用同一个公式。

        Args:
            image: 要显示的图（可能只是原图的一块）。
            scene_origin: 这张图的左上角在场景里的位置。
            scene_span: 这张图在场景里占多大。
            image_rect: 这张图覆盖原图的 ``(x0, y0, x1, y1)``。
            scene_size: 更换后的场景总尺寸。
            anchor_scene / anchor_image: 若给出，重设后让"原图上 anchor_image 这一点"
                仍停在原来的场景位置 anchor_scene，避免缩放时画面跳动。
        """
        self._image_size = (image.width(), image.height())
        self._scene_origin = (float(scene_origin[0]), float(scene_origin[1]))
        self._scene_span = (float(scene_span[0]), float(scene_span[1]))
        self._image_rect = tuple(float(v) for v in image_rect)  # type: ignore[assignment]
        self._pixmap = QPixmap.fromImage(image)
        self._item.setPixmap(self._pixmap)
        self._item.setPos(self._scene_origin[0], self._scene_origin[1])
        self._scene.setSceneRect(0, 0, float(scene_size[0]), float(scene_size[1]))
        if anchor_scene is not None and anchor_image is not None:
            ax, ay = self.to_scene(anchor_image[0], anchor_image[1])
            self.translate(anchor_scene[0] - ax, anchor_scene[1] - ay)

    def set_full_scene(self, image: QImage, original_size: tuple[int, int]) -> None:
        """回到"整幅图"显示（场景原点 0,0，覆盖整张原图）。"""
        self._image_size = (image.width(), image.height())
        self._original_size = original_size
        self._scene_origin = (0.0, 0.0)
        self._scene_span = (float(image.width()), float(image.height()))
        self._image_rect = (0.0, 0.0, float(original_size[0]), float(original_size[1]))
        self._pixmap = QPixmap.fromImage(image)
        self._item.setPixmap(self._pixmap)
        self._item.setPos(0.0, 0.0)
        self._scene.setSceneRect(0, 0, float(image.width()), float(image.height()))

    def set_view_size(self, view_size: tuple[float, float]) -> None:
        """保留接口（坐标换算已改为按"覆盖的原图区域"计算，不再需要整幅尺寸）。"""
        _ = view_size

    def set_mode(self, mode: str) -> None:
        """切换交互模式；切换时丢弃未完成的多边形。"""
        self.mode = mode
        self.cancel_polygon()
        cursors = {"pan": Qt.ArrowCursor, "roi": Qt.CrossCursor, "hint+": Qt.PointingHandCursor,
                   "hint-": Qt.PointingHandCursor}
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))

    def pending_polygon(self) -> list[tuple[float, float]]:
        return list(self._pending)

    def cancel_polygon(self) -> None:
        if self._pending:
            self._pending = []
            self.polygon_in_progress.emit([])

    def commit_polygon(self) -> None:
        """结束当前多边形；顶点数 >= 3 才提交。"""
        if len(self._pending) >= 3:
            self.polygon_committed.emit(list(self._pending))
        # 记录本次框选的原始数据，便于核对坐标换算
        self.last_polygon_debug = {
            "image_size": self._image_size,
            "original_size": self._original_size,
            "scale": self.image_to_display_scale(),
            "first_point_viewport": getattr(self, "_first_point_viewport", None),
            "points_scene": [tuple(round(v, 1) for v in p) for p in self._pending[:4]],
            "clicks": self._pending_clicks[:8],
        }
        self._pending = []
        self.polygon_in_progress.emit([])

    def fit_to_window(self) -> None:
        if self._pixmap.isNull():
            return
        self._user_viewed = False
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.zoom_changed.emit(self.current_scale())

    def resizeEvent(self, event) -> None:  # noqa: N802
        """窗口尺寸变化时重新适应。

        只在**使用者还没自己缩放过**时自动适应：一旦缩放/平移过，再强行
        ``fitInView`` 会把使用者的视角重置掉（表现为"放大后一动就跳回全图"）。
        """
        super().resizeEvent(event)
        if self._pixmap.isNull() or self._user_viewed:
            return
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def zoom_to(self, scale: float) -> None:
        self.resetTransform()
        self.scale(scale, scale)
        self._user_viewed = True
        self.zoom_changed.emit(self.current_scale())

    def current_scale(self) -> float:
        return float(self.transform().m11())

    def image_size(self) -> tuple[int, int]:
        return self._image_size

    # ------------------------------------------------------------------ 交互
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        new_scale = self.current_scale() * factor
        if self.MIN_ZOOM <= new_scale <= self.MAX_ZOOM:
            self.scale(factor, factor)
            self._user_viewed = True
            self.zoom_changed.emit(self.current_scale())
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pt = self.mapToScene(event.pos())
        ix, iy = self.to_image(float(pt.x()), float(pt.y()))

        if event.button() == Qt.RightButton:
            if self.mode == "roi":
                self.commit_polygon()
                event.accept()
                return
            if self.mode in ("hint+", "hint-"):
                self.hint_remove_requested.emit(ix, iy)
                event.accept()
                return
            self.right_clicked.emit(ix, iy)
            event.accept()
            return

        if event.button() in (Qt.MiddleButton,) or (
            event.button() == Qt.LeftButton and event.modifiers() & Qt.ShiftModifier
        ):
            self._panning = True
            self._pan_start = event.pos()
            self._user_viewed = True
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            if self.mode == "roi":
                if not self._pending:
                    # 记下本次框选的第一个"视口坐标"，用于核对换算
                    self._first_point_viewport = (event.pos().x(), event.pos().y())
                    self._first_point_scene = (float(pt.x()), float(pt.y()))
                self._pending.append((ix, iy))
                self._pending_clicks.append(
                    {
                        "viewport": (event.pos().x(), event.pos().y()),
                        "scene_raw": (round(float(pt.x()), 2), round(float(pt.y()), 2)),
                        "image": (round(ix, 2), round(iy, 2)),
                    }
                )
                self.polygon_in_progress.emit(list(self._pending))
                event.accept()
                return
            if not self._pending:
                self._first_point_viewport = (event.pos().x(), event.pos().y())
                self._first_point_scene = (float(pt.x()), float(pt.y()))
            self._last_click_debug = {
                "viewport": (event.pos().x(), event.pos().y()),
                "scene_raw": (round(float(pt.x()), 2), round(float(pt.y()), 2)),
                "image": (round(ix, 2), round(iy, 2)),
                "image_size": self._image_size,
                "original_size": self._original_size,
                "scale": round(self.image_to_display_scale(), 5),
                "m11": round(self.transform().m11(), 5),
            }
            self.clicked.emit(ix, iy)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        pt = self.mapToScene(event.pos())
        # cursor_moved / dragged 一律发**原图坐标**，与 clicked 保持一致
        ix, iy = self.to_image(float(pt.x()), float(pt.y()))
        self.cursor_moved.emit(ix, iy)
        if event.buttons() & Qt.LeftButton:
            self.dragged.emit(ix, iy)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning:
            self._panning = False
            self.set_mode(self.mode)
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self.drag_released.emit()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        # 双击只用于"结束当前多边形"；放大/缩小一律用滚轮。
        if self.mode == "roi" and event.button() == Qt.LeftButton:
            self.commit_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self.cancel_polygon()
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter) and self.mode == "roi":
            self.commit_polygon()
        else:
            super().keyPressEvent(event)

    def fit_on_next_show(self) -> None:
        self._fit_on_next = True

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._fit_on_next and not self._pixmap.isNull():
            self.fit_to_window()
            self._fit_on_next = False

