"use client";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@bluevoid/ui/card";
// import { ThemeToggle } from "@bluevoid/ui";
import { Toaster } from "@bluevoid/ui/sonner";
import { cn } from "@bluevoid/ui/utils";
// import { IconLoader2 } from "@tabler/icons-react";
import { type ComponentType, lazy, Suspense, useId } from "react";
import { ErrorBoundary } from "react-error-boundary";

// import {ThemeProvider} from "@bluevoid/ui/theme"

const TestAccordion = lazy(
	() => import("@/test-components/shadcn/test-accordion"),
);
const TestAlert = lazy(() => import("@/test-components/shadcn/test-alert"));
const TestAlertDialog = lazy(
	() => import("@/test-components/shadcn/test-alert-dialog"),
);
const TestAspectRatio = lazy(
	() => import("@/test-components/shadcn/test-aspect-ratio"),
);
const TestBadge = lazy(() => import("@/test-components/shadcn/test-badge"));
const TestBreadcrumb = lazy(
	() => import("@/test-components/shadcn/test-breadcrumb"),
);
const TestButton = lazy(() => import("@/test-components/shadcn/test-button"));
const TestCalendar = lazy(
	() => import("@/test-components/shadcn/test-calendar"),
);

const TestCard = lazy(() => import("@/test-components/shadcn/test-card"));
const TestCarousel = lazy(
	() => import("@/test-components/shadcn/test-carousel"),
);
const TestChart = lazy(() => import("@/test-components/shadcn/test-chart"));
const TestCheckbox = lazy(
	() => import("@/test-components/shadcn/test-checkbox"),
);
const TestCollapsible = lazy(
	() => import("@/test-components/shadcn/test-collapsible"),
);
const TestCombobox = lazy(
	() => import("@/test-components/shadcn/test-combobox"),
);
const TestCommand = lazy(() => import("@/test-components/shadcn/test-command"));
const TestContextMenu = lazy(
	() => import("@/test-components/shadcn/test-context-menu"),
);
const TestDialog = lazy(() => import("@/test-components/shadcn/test-dialog"));
const TestDrawer = lazy(() => import("@/test-components/shadcn/test-drawer"));
const TestDropdownMenu = lazy(
	() => import("@/test-components/shadcn/test-dropdown-menu"),
);
const TestForm = lazy(() => import("@/test-components/shadcn/test-form"));
const TestHoverCard = lazy(
	() => import("@/test-components/shadcn/test-hover-card"),
);
const TestInputOTP = lazy(
	() => import("@/test-components/shadcn/test-input-otp"),
);
const TestInput = lazy(() => import("@/test-components/shadcn/test-input"));
const TestLabel = lazy(() => import("@/test-components/shadcn/test-label"));
const TestMenubar = lazy(() => import("@/test-components/shadcn/test-menubar"));
const TestNavigationMenu = lazy(
	() => import("@/test-components/shadcn/test-navigation-menu"),
);
const TestPagination = lazy(
	() => import("@/test-components/shadcn/test-pagination"),
);
const TestPaginationPrimitive = lazy(
	() => import("@/test-components/shadcn/test-pagination-primitive"),
);
const TestPopover = lazy(() => import("@/test-components/shadcn/test-popover"));
const TestProgress = lazy(
	() => import("@/test-components/shadcn/test-progress"),
);
const TestRatioGroup = lazy(
	() => import("@/test-components/shadcn/test-ratio-group"),
);
const TestResizable = lazy(
	() => import("@/test-components/shadcn/test-resizable"),
);
const TestScrollArea = lazy(
	() => import("@/test-components/shadcn/test-scroll-area"),
);
const TestSelect = lazy(() => import("@/test-components/shadcn/test-select"));
const TestSeparator = lazy(
	() => import("@/test-components/shadcn/test-separator"),
);
const TestSheet = lazy(() => import("@/test-components/shadcn/test-sheet"));
const TestSkeleton = lazy(
	() => import("@/test-components/shadcn/test-skeleton"),
);
const TestSlider = lazy(() => import("@/test-components/shadcn/test-slider"));
const TestSonner = lazy(() => import("@/test-components/shadcn/test-sonner"));
const TestSwitch = lazy(() => import("@/test-components/shadcn/test-switch"));
const TestTable = lazy(() => import("@/test-components/shadcn/test-table"));
const TestTabs = lazy(() => import("@/test-components/shadcn/test-tabs"));
const TestTextarea = lazy(
	() => import("@/test-components/shadcn/test-textarea"),
);
const TestToggle = lazy(() => import("@/test-components/shadcn/test-toggle"));
const TestTooltip = lazy(() => import("@/test-components/shadcn/test-tooltip"));

const UIElements: {
	title: string;
	description?: string;
	Element: ComponentType;
	className?: string;
	disabled?: boolean;
}[] = [
	{
		title: "Accordion",
		Element: TestAccordion,
	},
	{
		title: "Alert",
		Element: TestAlert,
		className: "flex-col",
	},
	{
		title: "AlertDialog",
		Element: TestAlertDialog,
	},
	{
		title: "AspectRatio",
		description: "16 / 9",
		Element: TestAspectRatio,
	},
	{
		title: "Badge",
		Element: TestBadge,
	},
	{
		title: "Breadcrumb",
		Element: TestBreadcrumb,
	},
	{
		title: "Button",
		Element: TestButton,
		className: "flex-col",
	},
	{
		title: "Calendar",
		Element: TestCalendar,
	},
	{
		title: "Card",
		Element: TestCard,
	},
	{
		title: "Carousel",
		Element: TestCarousel,
		className: "justify-center",
	},
	{
		title: "Chart",
		Element: TestChart,
	},
	{
		title: "Checkbox",
		Element: TestCheckbox,
		className: "flex-col",
	},
	{
		title: "Collapsible",
		Element: TestCollapsible,
		className: "flex-col",
	},
	{
		title: "Combobox",
		Element: TestCombobox,
	},
	{
		title: "Command",
		Element: TestCommand,
	},
	{
		title: "ContextMenu",
		Element: TestContextMenu,
	},
	{
		title: "Dialog",
		Element: TestDialog,
	},
	{
		title: "Drawer",
		Element: TestDrawer,
	},
	{
		title: "DropdownMenu",
		Element: TestDropdownMenu,
		className: "justify-center",
	},

	{
		title: "Form",
		Element: TestForm,
	},
	{
		title: "HoverCard",
		Element: TestHoverCard,
		className: "justify-center",
	},
	{
		title: "InputOTP",
		Element: TestInputOTP,
		className: "flex-col",
	},
	{
		title: "Input",
		Element: TestInput,
		className: "flex-col",
	},
	{
		title: "Label",
		Element: TestLabel,
		className: "flex-col",
	},
	{
		title: "Menubar",
		Element: TestMenubar,
	},
	{
		title: "NavigationMenu",
		Element: TestNavigationMenu,
	},
	{
		title: "Pagination",
		Element: TestPagination,
		className: "flex-col",
	},
	{
		title: "Pagination Primitive",
		Element: TestPaginationPrimitive,
		className: "flex-col",
	},
	{
		title: "Popover",
		Element: TestPopover,
	},
	{
		title: "Progress",
		Element: TestProgress,
		// disabled: true,
	},
	{
		title: "RatioGroup",
		Element: TestRatioGroup,
	},
	{
		title: "Resizable",
		Element: TestResizable,
	},
	{
		title: "ScrollArea",
		Element: TestScrollArea,
	},
	{
		title: "Select",
		Element: TestSelect,
		className: "flex-col",
	},
	{
		title: "Separator",
		Element: TestSeparator,
	},
	{
		title: "Sheet",
		Element: TestSheet,
	},
	{
		title: "Skeleton",
		Element: TestSkeleton,
		className: "flex-col",
	},
	{
		title: "Slider",
		Element: TestSlider,
		className: "flex-col gap-7 pb-7",
	},
	{
		title: "Sonner",
		Element: TestSonner,
	},
	{
		title: "Switch",
		Element: TestSwitch,
	},
	{
		title: "Table",
		Element: TestTable,
	},
	{
		title: "Tabs",
		Element: TestTabs,
		className: "justify-center",
	},
	{
		title: "TextArea",
		Element: TestTextarea,
		className: "flex-col",
	},
	{
		title: "Toggle",
		Element: TestToggle,
		className: "flex-col",
	},
	{
		title: "Tooltip",
		Element: TestTooltip,
		className: "flex-col",
	},
];

function ShadCN() {
	// const { toggleTheme, theme } = useUserContext();
	const uuid = useId();
	console.log("test");
	return (
		<div className="mx-auto flex max-w-(--breakpoint-xl) flex-col gap-4 p-2 pb-96">
			{/* <TooltipProvider> */}
			{/* <ThemeToggle /> */}
			{UIElements.map((val, index) => (
				<Card key={`${uuid}${val.title}:${index}`}>
					<CardHeader>
						<CardTitle>{val.title}</CardTitle>
						{val.description !== undefined && (
							<CardDescription>{val.description}</CardDescription>
						)}
					</CardHeader>
					<CardContent className={cn("flex gap-2 p-2", val.className)}>
						{!val?.disabled ? (
							<ErrorBoundary onError={console.log} fallback={<div>Something went wrong</div>}>
								<Suspense
									fallback={
										"Loading..."
										// <IconLoader2 className="direction-reverse animate-spin" />
									}
								>
									<val.Element />
								</Suspense>
							</ErrorBoundary>
						) : (
							<div>Component disabled</div>
						)}
					</CardContent>
				</Card>
			))}
			<Toaster />
			{/* </TooltipProvider> */}
		</div>
	);
}

export default ShadCN;
